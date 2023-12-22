from __future__ import annotations



def forward(self, primals_1: "f32[768, 768]", primals_2: "f32[768]", primals_3: "f32[768, 768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[3072, 768]", primals_12: "f32[3072]", primals_13: "f32[768, 3072]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768, 768]", primals_18: "f32[768]", primals_19: "f32[768, 768]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[3072, 768]", primals_28: "f32[3072]", primals_29: "f32[768, 3072]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_35: "f32[768, 768]", primals_36: "f32[768]", primals_37: "f32[768, 768]", primals_38: "f32[768]", primals_39: "f32[768, 768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[3072, 768]", primals_44: "f32[3072]", primals_45: "f32[768, 3072]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768, 768]", primals_50: "f32[768]", primals_51: "f32[768, 768]", primals_52: "f32[768]", primals_53: "f32[768, 768]", primals_54: "f32[768]", primals_55: "f32[768, 768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[3072, 768]", primals_60: "f32[3072]", primals_61: "f32[768, 3072]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768, 768]", primals_66: "f32[768]", primals_67: "f32[768, 768]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768, 768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[3072, 768]", primals_76: "f32[3072]", primals_77: "f32[768, 3072]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768, 768]", primals_82: "f32[768]", primals_83: "f32[768, 768]", primals_84: "f32[768]", primals_85: "f32[768, 768]", primals_86: "f32[768]", primals_87: "f32[768, 768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[3072, 768]", primals_92: "f32[3072]", primals_93: "f32[768, 3072]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[768, 768]", primals_98: "f32[768]", primals_99: "f32[768, 768]", primals_100: "f32[768]", primals_101: "f32[768, 768]", primals_102: "f32[768]", primals_103: "f32[768, 768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[3072, 768]", primals_108: "f32[3072]", primals_109: "f32[768, 3072]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768, 768]", primals_114: "f32[768]", primals_115: "f32[768, 768]", primals_116: "f32[768]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768, 768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[3072, 768]", primals_124: "f32[3072]", primals_125: "f32[768, 3072]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_131: "f32[768, 768]", primals_132: "f32[768]", primals_133: "f32[768, 768]", primals_134: "f32[768]", primals_135: "f32[768, 768]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[3072, 768]", primals_140: "f32[3072]", primals_141: "f32[768, 3072]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768, 768]", primals_146: "f32[768]", primals_147: "f32[768, 768]", primals_148: "f32[768]", primals_149: "f32[768, 768]", primals_150: "f32[768]", primals_151: "f32[768, 768]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768]", primals_155: "f32[3072, 768]", primals_156: "f32[3072]", primals_157: "f32[768, 3072]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[768]", primals_161: "f32[768, 768]", primals_162: "f32[768]", primals_163: "f32[768, 768]", primals_164: "f32[768]", primals_165: "f32[768, 768]", primals_166: "f32[768]", primals_167: "f32[768, 768]", primals_168: "f32[768]", primals_169: "f32[768]", primals_170: "f32[768]", primals_171: "f32[3072, 768]", primals_172: "f32[3072]", primals_173: "f32[768, 3072]", primals_174: "f32[768]", primals_175: "f32[768]", primals_176: "f32[768]", primals_177: "f32[768, 768]", primals_178: "f32[768]", primals_179: "f32[768, 768]", primals_180: "f32[768]", primals_181: "f32[768, 768]", primals_182: "f32[768]", primals_183: "f32[768, 768]", primals_184: "f32[768]", primals_185: "f32[768]", primals_186: "f32[768]", primals_187: "f32[3072, 768]", primals_188: "f32[3072]", primals_189: "f32[768, 3072]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[1, 1024, 768]", primals_194: "f32[1, 1024]", primals_195: "b8[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view: "f32[1024, 768]" = torch.ops.aten.view.default(permute, [1024, 768]);  permute = None
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_2, view, permute_1);  primals_2 = None
    view_1: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm, [1024, 1, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_4, view, permute_2);  primals_4 = None
    view_3: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_1, [1024, 1, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_6, view, permute_3);  primals_6 = None
    view_5: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_2, [1024, 1, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_1, 8.0);  view_1 = None
    view_6: "f32[1024, 768]" = torch.ops.aten.view.default(div, [1024, 768]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_9: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_3, [1024, 1, 12, 64]);  view_3 = None
    permute_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_9, [1, 0, 2, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_7, [12, 1024, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_13: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_11, [12, 2, 512, 64]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_1: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_13, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_1: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
    permute_9: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 3]);  unsqueeze_1 = None
    view_14: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_6, [1024, 1, 768]);  view_6 = None
    view_15: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_14, [1024, 1, 12, 64]);  view_14 = None
    permute_11: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_15, [1, 0, 2, 3]);  view_15 = None
    permute_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_11, [0, 2, 1, 3]);  permute_11 = None
    view_16: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_12, [12, 1024, 64]);  permute_12 = None
    view_17: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_16, [12, 2, 512, 64]);  view_16 = None
    as_strided_2: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_17, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_17 = None
    unsqueeze_2: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
    permute_13: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 2, 4, 3]);  unsqueeze_2 = None
    permute_14: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_13, [0, 1, 2, 4, 3]);  permute_13 = None
    clone: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_18: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone, [36, 512, 64]);  clone = None
    permute_15: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_9, [0, 1, 4, 3, 2]);  permute_9 = None
    clone_1: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_19: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_1, [36, 64, 512]);  clone_1 = None
    bmm: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_18, view_19)
    view_20: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, [12, 3, 512, 1, 512]);  bmm = None
    permute_16: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_20, [0, 1, 2, 4, 3]);  view_20 = None
    view_21: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_16, [12, 3, 512, 512]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_21, [0, 0, 0, 1], 0.0);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_22: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, [12, 3, 512, 513]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_22, 0, 0, 9223372036854775807);  view_22 = None
    slice_2: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807)
    slice_3: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 256)
    slice_4: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257);  slice_3 = None
    slice_5: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_6: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, -1)
    slice_7: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807)
    slice_8: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_7, 3, 256, 9223372036854775807)
    copy: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_4);  slice_4 = None
    slice_scatter: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy, 3, 256, 9223372036854775807);  copy = None
    slice_scatter_1: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter, 2, 0, 9223372036854775807);  slice_scatter = None
    slice_scatter_2: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_1, 1, 0, -1);  slice_scatter_1 = None
    slice_scatter_3: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_2, 0, 0, 9223372036854775807);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1, 1, -1)
    slice_17: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
    slice_18: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 257);  slice_17 = None
    slice_22: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 9223372036854775807)
    select_2: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_22, 1, -1)
    slice_23: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 0, 9223372036854775807)
    slice_24: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_23, 2, 256, 9223372036854775807)
    copy_1: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_24, slice_18);  slice_24 = slice_18 = None
    slice_scatter_4: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_23, copy_1, 2, 256, 9223372036854775807);  slice_23 = copy_1 = None
    slice_scatter_5: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_2, slice_scatter_4, 1, 0, 9223372036854775807);  select_2 = slice_scatter_4 = None
    select_scatter: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_22, slice_scatter_5, 1, -1);  slice_22 = slice_scatter_5 = None
    slice_scatter_6: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_3 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_32: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2, 2, -257, -1);  slice_2 = None
    slice_33: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_32, 3, 257, 9223372036854775807);  slice_32 = None
    slice_38: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 0, 9223372036854775807)
    slice_39: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_38, 1, 1, 9223372036854775807)
    slice_40: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_39, 2, 0, 9223372036854775807)
    slice_41: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 256)
    copy_2: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_41, slice_33);  slice_41 = slice_33 = None
    slice_scatter_7: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_40, copy_2, 3, 0, 256);  slice_40 = copy_2 = None
    slice_scatter_8: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_39, slice_scatter_7, 2, 0, 9223372036854775807);  slice_39 = slice_scatter_7 = None
    slice_scatter_9: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_38, slice_scatter_8, 1, 1, 9223372036854775807);  slice_38 = slice_scatter_8 = None
    slice_scatter_10: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_6, slice_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_6 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_5: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
    slice_50: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
    slice_51: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_50, 2, -255, 9223372036854775807);  slice_50 = None
    slice_55: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 0, 9223372036854775807)
    select_7: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_55, 1, 0)
    slice_56: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256)
    slice_57: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_56, 2, 1, 256)
    copy_3: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_57, slice_51);  slice_57 = slice_51 = None
    slice_scatter_11: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_56, copy_3, 2, 1, 256);  slice_56 = copy_3 = None
    slice_scatter_12: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_7, slice_scatter_11, 1, 1, 256);  select_7 = slice_scatter_11 = None
    select_scatter_1: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_55, slice_scatter_12, 1, 0);  slice_55 = slice_scatter_12 = None
    slice_scatter_13: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_10, select_scatter_1, 0, 0, 9223372036854775807);  slice_scatter_10 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_default: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_3: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_4: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
    le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[256, 257]" = torch.ops.aten.where.self(le, full_default, full_default_1);  le = full_default = None
    rev: "f32[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_5: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
    slice_63: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_5, 1, 0, 9223372036854775807);  unsqueeze_5 = None
    unsqueeze_6: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_63, 2);  slice_63 = None
    slice_64: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 3, 0, 9223372036854775807);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_1: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_64, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_64, [1, 256, 12, 257])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_25: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_13, [1, 12, 1024, 513]);  slice_scatter_13 = None
    permute_19: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    slice_69: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_19, 0, 0, 9223372036854775807)
    slice_70: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_69, 1, 0, 256)
    slice_71: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, 9223372036854775807)
    slice_72: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 257)
    full_default_2: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
    where_1: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_4: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_72, where_1);  slice_72 = where_1 = None
    slice_scatter_14: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_71, copy_4, 3, 0, 257);  slice_71 = copy_4 = None
    slice_scatter_15: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_70, slice_scatter_14, 2, 0, 9223372036854775807);  slice_70 = slice_scatter_14 = None
    slice_scatter_16: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_69, slice_scatter_15, 1, 0, 256);  slice_69 = slice_scatter_15 = None
    slice_scatter_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_19, slice_scatter_16, 0, 0, 9223372036854775807);  permute_19 = slice_scatter_16 = None
    permute_22: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_17, [0, 2, 1, 3]);  slice_scatter_17 = None
    view_28: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_22, [12, 4, 256, 513]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_1: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [1, 256, 12, 257])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_30: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_28, [1, 12, 1024, 513]);  view_28 = None
    permute_24: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    slice_92: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_24, 0, 0, 9223372036854775807)
    slice_93: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_92, 1, -256, 9223372036854775807)
    slice_94: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_93, 2, 0, 9223372036854775807)
    slice_95: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_94, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_1: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
    where_2: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_5: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_95, where_2);  slice_95 = where_2 = None
    slice_scatter_18: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_94, copy_5, 3, -257, 9223372036854775807);  slice_94 = copy_5 = None
    slice_scatter_19: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_93, slice_scatter_18, 2, 0, 9223372036854775807);  slice_93 = slice_scatter_18 = None
    slice_scatter_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_92, slice_scatter_19, 1, -256, 9223372036854775807);  slice_92 = slice_scatter_19 = None
    slice_scatter_21: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_24, slice_scatter_20, 0, 0, 9223372036854775807);  permute_24 = slice_scatter_20 = None
    permute_27: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None
    view_33: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_27, [12, 4, 256, 513]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(primals_194, 0);  primals_194 = None
    slice_111: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne, 0, 0, 9223372036854775807);  ne = None
    slice_112: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_111, 1, 0, 9223372036854775807);  slice_111 = None
    unsqueeze_7: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_112, 2);  slice_112 = None
    unsqueeze_8: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_2: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_3: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_8, full_default_4, convert_element_type_2);  unsqueeze_8 = full_default_4 = convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_4: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_29: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
    view_35: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_29, [1, 1024, 1]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_30: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
    view_36: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_30, [1, 1024, 1]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_37: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_35, [1, 2, 512, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_3: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_37, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_38: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_36, [1, 2, 512, 1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_4: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_38, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_9: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
    permute_31: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 2, 4, 3]);  unsqueeze_9 = None
    unsqueeze_10: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
    permute_32: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 4, 2, 3]);  unsqueeze_10 = None
    mul: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_31, permute_32);  permute_31 = permute_32 = None
    view_39: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul, [1, 3, 512, 512]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_1: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_39, [0, 0, 0, 1], 0.0);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_40: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, [1, 3, 512, 513]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_5: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_113: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_40, 0, 0, 9223372036854775807);  view_40 = None
    slice_114: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_113, 1, 0, 9223372036854775807)
    slice_115: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_114, 2, 0, 256)
    slice_116: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 257);  slice_115 = None
    slice_117: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_118: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_117, 1, 0, -1)
    slice_119: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_118, 2, 0, 9223372036854775807)
    slice_120: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_119, 3, 256, 9223372036854775807)
    copy_6: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_120, slice_116);  slice_120 = slice_116 = None
    slice_scatter_22: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_119, copy_6, 3, 256, 9223372036854775807);  slice_119 = copy_6 = None
    slice_scatter_23: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_118, slice_scatter_22, 2, 0, 9223372036854775807);  slice_118 = slice_scatter_22 = None
    slice_scatter_24: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_117, slice_scatter_23, 1, 0, -1);  slice_117 = slice_scatter_23 = None
    slice_scatter_25: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_24, 0, 0, 9223372036854775807);  full_5 = slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_10: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_113, 1, -1)
    slice_129: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 256, 9223372036854775807);  select_10 = None
    slice_130: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_129, 2, 0, 257);  slice_129 = None
    slice_134: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 0, 9223372036854775807)
    select_12: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_134, 1, -1)
    slice_135: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_12, 1, 0, 9223372036854775807)
    slice_136: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_135, 2, 256, 9223372036854775807)
    copy_7: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_136, slice_130);  slice_136 = slice_130 = None
    slice_scatter_26: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_135, copy_7, 2, 256, 9223372036854775807);  slice_135 = copy_7 = None
    slice_scatter_27: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_12, slice_scatter_26, 1, 0, 9223372036854775807);  select_12 = slice_scatter_26 = None
    select_scatter_2: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_134, slice_scatter_27, 1, -1);  slice_134 = slice_scatter_27 = None
    slice_scatter_28: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_25, select_scatter_2, 0, 0, 9223372036854775807);  slice_scatter_25 = select_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_144: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_114, 2, -257, -1);  slice_114 = None
    slice_145: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_144, 3, 257, 9223372036854775807);  slice_144 = None
    slice_150: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 0, 0, 9223372036854775807)
    slice_151: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_150, 1, 1, 9223372036854775807)
    slice_152: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_151, 2, 0, 9223372036854775807)
    slice_153: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_152, 3, 0, 256)
    copy_8: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_153, slice_145);  slice_153 = slice_145 = None
    slice_scatter_29: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_152, copy_8, 3, 0, 256);  slice_152 = copy_8 = None
    slice_scatter_30: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_151, slice_scatter_29, 2, 0, 9223372036854775807);  slice_151 = slice_scatter_29 = None
    slice_scatter_31: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_150, slice_scatter_30, 1, 1, 9223372036854775807);  slice_150 = slice_scatter_30 = None
    slice_scatter_32: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_28, slice_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_28 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_15: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_113, 1, 0);  slice_113 = None
    slice_162: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_15, 1, 0, 255);  select_15 = None
    slice_163: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_162, 2, -255, 9223372036854775807);  slice_162 = None
    slice_167: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    select_17: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_167, 1, 0)
    slice_168: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_17, 1, 1, 256)
    slice_169: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_168, 2, 1, 256)
    copy_9: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_169, slice_163);  slice_169 = slice_163 = None
    slice_scatter_33: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_168, copy_9, 2, 1, 256);  slice_168 = copy_9 = None
    slice_scatter_34: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_17, slice_scatter_33, 1, 1, 256);  select_17 = slice_scatter_33 = None
    select_scatter_3: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_167, slice_scatter_34, 1, 0);  slice_167 = slice_scatter_34 = None
    slice_scatter_35: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, select_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_32 = select_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_2: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_64, [1, 256, 1, 257])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_43: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_35, [1, 1, 1024, 513]);  slice_scatter_35 = None
    permute_35: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    slice_181: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_35, 0, 0, 9223372036854775807)
    slice_182: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_181, 1, 0, 256)
    slice_183: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_182, 2, 0, 9223372036854775807)
    slice_184: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_183, 3, 0, 257)
    full_default_7: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_3: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
    where_5: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_default_7, slice_184);  convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_10: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_184, where_5);  slice_184 = where_5 = None
    slice_scatter_36: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_183, copy_10, 3, 0, 257);  slice_183 = copy_10 = None
    slice_scatter_37: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_182, slice_scatter_36, 2, 0, 9223372036854775807);  slice_182 = slice_scatter_36 = None
    slice_scatter_38: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_181, slice_scatter_37, 1, 0, 256);  slice_181 = slice_scatter_37 = None
    slice_scatter_39: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_35, slice_scatter_38, 0, 0, 9223372036854775807);  permute_35 = slice_scatter_38 = None
    permute_38: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_39, [0, 2, 1, 3]);  slice_scatter_39 = None
    view_46: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_38, [1, 4, 256, 513]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_3: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_1, [1, 256, 1, 257])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_48: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_46, [1, 1, 1024, 513]);  view_46 = None
    permute_40: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    slice_204: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_40, 0, 0, 9223372036854775807)
    slice_205: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_204, 1, -256, 9223372036854775807)
    slice_206: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_205, 2, 0, 9223372036854775807)
    slice_207: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_206, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_4: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
    where_6: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_default_7, slice_207);  convert_element_type_4 = full_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_11: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_207, where_6);  slice_207 = where_6 = None
    slice_scatter_40: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_206, copy_11, 3, -257, 9223372036854775807);  slice_206 = copy_11 = None
    slice_scatter_41: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_205, slice_scatter_40, 2, 0, 9223372036854775807);  slice_205 = slice_scatter_40 = None
    slice_scatter_42: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_204, slice_scatter_41, 1, -256, 9223372036854775807);  slice_204 = slice_scatter_41 = None
    slice_scatter_43: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_40, slice_scatter_42, 0, 0, 9223372036854775807);  permute_40 = slice_scatter_42 = None
    permute_43: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None
    view_51: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_43, [1, 4, 256, 513]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_53: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_33, [1, 12, 1024, 513]);  view_33 = None
    permute_45: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    view_54: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_51, [1, 1, 1024, 513]);  view_51 = None
    permute_46: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    add_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_45, permute_46);  permute_45 = None
    permute_47: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_2, [0, 2, 1, 3]);  add_2 = None
    view_56: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_47, [12, 4, 256, 513]);  permute_47 = None
    view_57: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_56, [1, 12, 1024, 513]);  view_56 = None
    permute_48: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_2, [-1], True)
    sub_4: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_2, amax);  clone_2 = amax = None
    exp: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_223: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(primals_195, 0, 0, 9223372036854775807);  primals_195 = None
    slice_224: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_223, 1, 0, 9223372036854775807);  slice_223 = None
    unsqueeze_15: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_224, 2);  slice_224 = None
    unsqueeze_16: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 3);  unsqueeze_15 = None
    where_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(where_7, 0.1, True);  where_7 = None
    getitem: "f32[1, 1024, 12, 513]" = native_dropout[0]
    getitem_1: "b8[1, 1024, 12, 513]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_58: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_5, [1024, 1, 12, 64]);  view_5 = None
    permute_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_58, [1, 0, 2, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_50: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
    view_59: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_50, [12, 4, 256, 513]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_51: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_49, [0, 2, 1, 3]);  permute_49 = None
    view_60: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_51, [12, 1024, 64]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_2: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_60, [0, 0, 256, 256], -1.0);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_5: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_3: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_59, [0, 257], 0.0);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_61: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_3, [12, 4, -1]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_225: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_61, 0, 0, 9223372036854775807);  view_61 = None
    slice_226: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_225, 1, 0, 9223372036854775807);  slice_225 = None
    slice_227: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_226, 2, 0, -256);  slice_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_62: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_227, [12, 4, 256, 769]);  slice_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_228: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_62, 0, 0, 9223372036854775807);  view_62 = None
    slice_229: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_228, 1, 0, 9223372036854775807);  slice_228 = None
    slice_230: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_229, 2, 0, 9223372036854775807);  slice_229 = None
    slice_231: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_230, 3, 0, -1);  slice_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_17: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_231, 4);  slice_231 = None
    permute_52: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [0, 1, 2, 4, 3]);  unsqueeze_17 = None
    unsqueeze_18: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
    permute_53: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_18, [0, 1, 4, 3, 2]);  unsqueeze_18 = None
    permute_54: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_52, [0, 1, 2, 4, 3]);  permute_52 = None
    view_63: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_54, [48, 256, 768]);  permute_54 = None
    permute_55: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_53, [0, 1, 4, 3, 2]);  permute_53 = None
    clone_3: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_64: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_3, [48, 768, 64]);  clone_3 = None
    bmm_1: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_63, view_64)
    view_65: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_1, [12, 4, 256, 1, 64]);  bmm_1 = None
    permute_56: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_65, [0, 1, 2, 4, 3]);  view_65 = None
    view_66: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_56, [12, 4, 256, 64]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_67: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_66, [1, 12, 1024, 64]);  view_66 = None
    permute_57: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_58: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_57, [1, 0, 2, 3]);  permute_57 = None
    clone_4: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_68: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_4, [1024, 1, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_59: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[1024, 768]" = torch.ops.aten.view.default(permute_59, [1024, 768]);  permute_59 = None
    permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_8, view_69, permute_60);  primals_8 = None
    view_70: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_70, 0.1, True);  view_70 = None
    getitem_2: "f32[1, 1024, 768]" = native_dropout_1[0]
    getitem_3: "b8[1, 1024, 768]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_2, primals_193);  getitem_2 = primals_193 = None
    var_mean = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_5);  add_4 = getitem_5 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_71: "f32[1024, 768]" = torch.ops.aten.view.default(add_6, [1024, 768])
    permute_61: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_4: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_12, view_71, permute_61);  primals_12 = None
    view_72: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_3: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, 0.5)
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476);  view_72 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_7);  mul_3 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_73: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_5, [1024, 3072]);  mul_5 = None
    permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_14, view_73, permute_62);  primals_14 = None
    view_74: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_74, 0.1, True);  view_74 = None
    getitem_6: "f32[1, 1024, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 1024, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_6, add_6);  getitem_6 = add_6 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_7: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
    mul_6: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = None
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, primals_15)
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_16);  mul_7 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_63: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_10, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_75: "f32[1024, 768]" = torch.ops.aten.view.default(permute_63, [1024, 768]);  permute_63 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_6: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_18, view_75, permute_64);  primals_18 = None
    view_76: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_6, [1024, 1, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_65: "f32[768, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_20, view_75, permute_65);  primals_20 = None
    view_78: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_7, [1024, 1, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_8: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_22, view_75, permute_66);  primals_22 = None
    view_80: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_8, [1024, 1, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_10: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_76, 8.0);  view_76 = None
    view_81: "f32[1024, 768]" = torch.ops.aten.view.default(div_10, [1024, 768]);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_84: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_78, [1024, 1, 12, 64]);  view_78 = None
    permute_68: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [1, 0, 2, 3]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_70: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_68, [0, 2, 1, 3]);  permute_68 = None
    view_86: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_70, [12, 1024, 64]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_88: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_86, [12, 2, 512, 64]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_7: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_88, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_20: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
    permute_72: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [0, 1, 4, 2, 3]);  unsqueeze_20 = None
    view_89: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_81, [1024, 1, 768]);  view_81 = None
    view_90: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_89, [1024, 1, 12, 64]);  view_89 = None
    permute_74: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_90, [1, 0, 2, 3]);  view_90 = None
    permute_75: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_74, [0, 2, 1, 3]);  permute_74 = None
    view_91: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_75, [12, 1024, 64]);  permute_75 = None
    view_92: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_91, [12, 2, 512, 64]);  view_91 = None
    as_strided_8: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_92, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_92 = None
    unsqueeze_21: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
    permute_76: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [0, 1, 2, 4, 3]);  unsqueeze_21 = None
    permute_77: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_76, [0, 1, 2, 4, 3]);  permute_76 = None
    clone_5: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_93: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_5, [36, 512, 64]);  clone_5 = None
    permute_78: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_72, [0, 1, 4, 3, 2]);  permute_72 = None
    clone_6: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_94: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_6, [36, 64, 512]);  clone_6 = None
    bmm_2: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_93, view_94)
    view_95: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_2, [12, 3, 512, 1, 512]);  bmm_2 = None
    permute_79: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_95, [0, 1, 2, 4, 3]);  view_95 = None
    view_96: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_79, [12, 3, 512, 512]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_4: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_96, [0, 0, 0, 1], 0.0);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_97: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_4, [12, 3, 512, 513]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_232: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_97, 0, 0, 9223372036854775807);  view_97 = None
    slice_233: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_232, 1, 0, 9223372036854775807)
    slice_234: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_233, 2, 0, 256)
    slice_235: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_234, 3, 0, 257);  slice_234 = None
    copy_12: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_235);  slice_235 = None
    slice_scatter_44: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_12, 3, 256, 9223372036854775807);  copy_12 = None
    slice_scatter_45: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_44, 2, 0, 9223372036854775807);  slice_scatter_44 = None
    slice_scatter_46: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_45, 1, 0, -1);  slice_scatter_45 = None
    slice_scatter_47: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_46, 0, 0, 9223372036854775807);  slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_20: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_232, 1, -1)
    slice_248: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
    slice_249: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 257);  slice_248 = None
    slice_253: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    select_22: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_253, 1, -1)
    slice_254: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_22, 1, 0, 9223372036854775807)
    slice_255: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_254, 2, 256, 9223372036854775807)
    copy_13: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_255, slice_249);  slice_255 = slice_249 = None
    slice_scatter_48: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_254, copy_13, 2, 256, 9223372036854775807);  slice_254 = copy_13 = None
    slice_scatter_49: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_22, slice_scatter_48, 1, 0, 9223372036854775807);  select_22 = slice_scatter_48 = None
    select_scatter_4: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_253, slice_scatter_49, 1, -1);  slice_253 = slice_scatter_49 = None
    slice_scatter_50: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_47, select_scatter_4, 0, 0, 9223372036854775807);  slice_scatter_47 = select_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_263: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_233, 2, -257, -1);  slice_233 = None
    slice_264: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_263, 3, 257, 9223372036854775807);  slice_263 = None
    slice_269: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_270: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_269, 1, 1, 9223372036854775807)
    slice_271: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_270, 2, 0, 9223372036854775807)
    slice_272: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_271, 3, 0, 256)
    copy_14: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_272, slice_264);  slice_272 = slice_264 = None
    slice_scatter_51: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_271, copy_14, 3, 0, 256);  slice_271 = copy_14 = None
    slice_scatter_52: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_270, slice_scatter_51, 2, 0, 9223372036854775807);  slice_270 = slice_scatter_51 = None
    slice_scatter_53: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_269, slice_scatter_52, 1, 1, 9223372036854775807);  slice_269 = slice_scatter_52 = None
    slice_scatter_54: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_53, 0, 0, 9223372036854775807);  slice_scatter_50 = slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_25: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_232, 1, 0);  slice_232 = None
    slice_281: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_25, 1, 0, 255);  select_25 = None
    slice_282: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_281, 2, -255, 9223372036854775807);  slice_281 = None
    slice_286: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_54, 0, 0, 9223372036854775807)
    select_27: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_286, 1, 0)
    slice_287: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256)
    slice_288: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_287, 2, 1, 256)
    copy_15: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_288, slice_282);  slice_288 = slice_282 = None
    slice_scatter_55: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_287, copy_15, 2, 1, 256);  slice_287 = copy_15 = None
    slice_scatter_56: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_27, slice_scatter_55, 1, 1, 256);  select_27 = slice_scatter_55 = None
    select_scatter_5: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_286, slice_scatter_56, 1, 0);  slice_286 = slice_scatter_56 = None
    slice_scatter_57: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_54, select_scatter_5, 0, 0, 9223372036854775807);  slice_scatter_54 = select_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_100: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_57, [1, 12, 1024, 513]);  slice_scatter_57 = None
    permute_82: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    slice_300: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_82, 0, 0, 9223372036854775807)
    slice_301: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_300, 1, 0, 256)
    slice_302: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_301, 2, 0, 9223372036854775807)
    slice_303: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_302, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_9: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_303)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_16: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_303, where_9);  slice_303 = where_9 = None
    slice_scatter_58: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_302, copy_16, 3, 0, 257);  slice_302 = copy_16 = None
    slice_scatter_59: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_301, slice_scatter_58, 2, 0, 9223372036854775807);  slice_301 = slice_scatter_58 = None
    slice_scatter_60: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_300, slice_scatter_59, 1, 0, 256);  slice_300 = slice_scatter_59 = None
    slice_scatter_61: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_82, slice_scatter_60, 0, 0, 9223372036854775807);  permute_82 = slice_scatter_60 = None
    permute_85: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_61, [0, 2, 1, 3]);  slice_scatter_61 = None
    view_103: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_85, [12, 4, 256, 513]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_105: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_103, [1, 12, 1024, 513]);  view_103 = None
    permute_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    slice_323: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_87, 0, 0, 9223372036854775807)
    slice_324: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_323, 1, -256, 9223372036854775807)
    slice_325: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_324, 2, 0, 9223372036854775807)
    slice_326: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_325, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_10: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_326)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_17: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_326, where_10);  slice_326 = where_10 = None
    slice_scatter_62: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_325, copy_17, 3, -257, 9223372036854775807);  slice_325 = copy_17 = None
    slice_scatter_63: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_324, slice_scatter_62, 2, 0, 9223372036854775807);  slice_324 = slice_scatter_62 = None
    slice_scatter_64: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_323, slice_scatter_63, 1, -256, 9223372036854775807);  slice_323 = slice_scatter_63 = None
    slice_scatter_65: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_87, slice_scatter_64, 0, 0, 9223372036854775807);  permute_87 = slice_scatter_64 = None
    permute_90: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None
    view_108: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_90, [12, 4, 256, 513]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_128: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_108, [1, 12, 1024, 513]);  view_108 = None
    permute_108: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    add_13: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_108, permute_46);  permute_108 = None
    permute_110: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_13, [0, 2, 1, 3]);  add_13 = None
    view_131: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_110, [12, 4, 256, 513]);  permute_110 = None
    view_132: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_131, [1, 12, 1024, 513]);  view_131 = None
    permute_111: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    amax_1: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_7, [-1], True)
    sub_12: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_7, amax_1);  clone_7 = amax_1 = None
    exp_1: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_2: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_15: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_3 = torch.ops.aten.native_dropout.default(where_15, 0.1, True);  where_15 = None
    getitem_10: "f32[1, 1024, 12, 513]" = native_dropout_3[0]
    getitem_11: "b8[1, 1024, 12, 513]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_133: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_80, [1024, 1, 12, 64]);  view_80 = None
    permute_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_133, [1, 0, 2, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_113: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
    view_134: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_113, [12, 4, 256, 513]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_114: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1, 3]);  permute_112 = None
    view_135: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_114, [12, 1024, 64]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_6: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, 0, 256, 256], -1.0);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_11: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_7: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_134, [0, 257], 0.0);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_136: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_7, [12, 4, -1]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_456: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_136, 0, 0, 9223372036854775807);  view_136 = None
    slice_457: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_456, 1, 0, 9223372036854775807);  slice_456 = None
    slice_458: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_457, 2, 0, -256);  slice_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_137: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_458, [12, 4, 256, 769]);  slice_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_459: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_137, 0, 0, 9223372036854775807);  view_137 = None
    slice_460: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_459, 1, 0, 9223372036854775807);  slice_459 = None
    slice_461: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_460, 2, 0, 9223372036854775807);  slice_460 = None
    slice_462: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_461, 3, 0, -1);  slice_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_36: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_462, 4);  slice_462 = None
    permute_115: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_36, [0, 1, 2, 4, 3]);  unsqueeze_36 = None
    unsqueeze_37: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
    permute_116: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 4, 3, 2]);  unsqueeze_37 = None
    permute_117: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_115, [0, 1, 2, 4, 3]);  permute_115 = None
    view_138: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_117, [48, 256, 768]);  permute_117 = None
    permute_118: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_116, [0, 1, 4, 3, 2]);  permute_116 = None
    clone_8: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_139: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_8, [48, 768, 64]);  clone_8 = None
    bmm_3: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_138, view_139)
    view_140: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_3, [12, 4, 256, 1, 64]);  bmm_3 = None
    permute_119: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_140, [0, 1, 2, 4, 3]);  view_140 = None
    view_141: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_119, [12, 4, 256, 64]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_142: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_141, [1, 12, 1024, 64]);  view_141 = None
    permute_120: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_121: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_120, [1, 0, 2, 3]);  permute_120 = None
    clone_9: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_143: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_9, [1024, 1, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_122: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_144: "f32[1024, 768]" = torch.ops.aten.view.default(permute_122, [1024, 768]);  permute_122 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_24, view_144, permute_123);  primals_24 = None
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_145, 0.1, True);  view_145 = None
    getitem_12: "f32[1, 1024, 768]" = native_dropout_4[0]
    getitem_13: "b8[1, 1024, 768]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_12, add_10);  getitem_12 = add_10 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_16: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_15);  add_15 = getitem_15 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_25)
    add_17: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_26);  mul_10 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_146: "f32[1024, 768]" = torch.ops.aten.view.default(add_17, [1024, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_28, view_146, permute_124);  primals_28 = None
    view_147: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, 0.5)
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, 0.7071067811865476);  view_147 = None
    erf_1: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_18: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_18);  mul_11 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_13, [1024, 3072]);  mul_13 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_30, view_148, permute_125);  primals_30 = None
    view_149: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_16: "f32[1, 1024, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 1024, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_17);  getitem_16 = add_17 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_19);  add_19 = getitem_19 = None
    mul_14: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = None
    mul_15: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_31)
    add_21: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_32);  mul_15 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_126: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_21, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_150: "f32[1024, 768]" = torch.ops.aten.view.default(permute_126, [1024, 768]);  permute_126 = None
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_12: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_34, view_150, permute_127);  primals_34 = None
    view_151: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_12, [1024, 1, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_36, view_150, permute_128);  primals_36 = None
    view_153: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_13, [1024, 1, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_14: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_38, view_150, permute_129);  primals_38 = None
    view_155: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_14, [1024, 1, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_20: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_151, 8.0);  view_151 = None
    view_156: "f32[1024, 768]" = torch.ops.aten.view.default(div_20, [1024, 768]);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_159: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_153, [1024, 1, 12, 64]);  view_153 = None
    permute_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_159, [1, 0, 2, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_133: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_131, [0, 2, 1, 3]);  permute_131 = None
    view_161: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_133, [12, 1024, 64]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_163: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_161, [12, 2, 512, 64]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_13: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_163, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_39: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
    permute_135: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_39, [0, 1, 4, 2, 3]);  unsqueeze_39 = None
    view_164: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_156, [1024, 1, 768]);  view_156 = None
    view_165: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_164, [1024, 1, 12, 64]);  view_164 = None
    permute_137: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_165, [1, 0, 2, 3]);  view_165 = None
    permute_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_137, [0, 2, 1, 3]);  permute_137 = None
    view_166: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_138, [12, 1024, 64]);  permute_138 = None
    view_167: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_166, [12, 2, 512, 64]);  view_166 = None
    as_strided_14: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_167, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_167 = None
    unsqueeze_40: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
    permute_139: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_40, [0, 1, 2, 4, 3]);  unsqueeze_40 = None
    permute_140: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_139, [0, 1, 2, 4, 3]);  permute_139 = None
    clone_10: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_168: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_10, [36, 512, 64]);  clone_10 = None
    permute_141: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_135, [0, 1, 4, 3, 2]);  permute_135 = None
    clone_11: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    view_169: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_11, [36, 64, 512]);  clone_11 = None
    bmm_4: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_4, [12, 3, 512, 1, 512]);  bmm_4 = None
    permute_142: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
    view_171: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_142, [12, 3, 512, 512]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_8: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_171, [0, 0, 0, 1], 0.0);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_172: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_8, [12, 3, 512, 513]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_463: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_172, 0, 0, 9223372036854775807);  view_172 = None
    slice_464: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_463, 1, 0, 9223372036854775807)
    slice_465: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_464, 2, 0, 256)
    slice_466: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_465, 3, 0, 257);  slice_465 = None
    copy_24: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_466);  slice_466 = None
    slice_scatter_88: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_24, 3, 256, 9223372036854775807);  copy_24 = None
    slice_scatter_89: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_88, 2, 0, 9223372036854775807);  slice_scatter_88 = None
    slice_scatter_90: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_89, 1, 0, -1);  slice_scatter_89 = None
    slice_scatter_91: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_90, 0, 0, 9223372036854775807);  slice_scatter_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_40: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_463, 1, -1)
    slice_479: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_40, 1, 256, 9223372036854775807);  select_40 = None
    slice_480: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_479, 2, 0, 257);  slice_479 = None
    slice_484: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 0, 0, 9223372036854775807)
    select_42: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_484, 1, -1)
    slice_485: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_42, 1, 0, 9223372036854775807)
    slice_486: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_485, 2, 256, 9223372036854775807)
    copy_25: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_486, slice_480);  slice_486 = slice_480 = None
    slice_scatter_92: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_485, copy_25, 2, 256, 9223372036854775807);  slice_485 = copy_25 = None
    slice_scatter_93: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_42, slice_scatter_92, 1, 0, 9223372036854775807);  select_42 = slice_scatter_92 = None
    select_scatter_8: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_484, slice_scatter_93, 1, -1);  slice_484 = slice_scatter_93 = None
    slice_scatter_94: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_91, select_scatter_8, 0, 0, 9223372036854775807);  slice_scatter_91 = select_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_494: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_464, 2, -257, -1);  slice_464 = None
    slice_495: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_494, 3, 257, 9223372036854775807);  slice_494 = None
    slice_500: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 0, 9223372036854775807)
    slice_501: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_500, 1, 1, 9223372036854775807)
    slice_502: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_501, 2, 0, 9223372036854775807)
    slice_503: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_502, 3, 0, 256)
    copy_26: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_503, slice_495);  slice_503 = slice_495 = None
    slice_scatter_95: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_502, copy_26, 3, 0, 256);  slice_502 = copy_26 = None
    slice_scatter_96: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_501, slice_scatter_95, 2, 0, 9223372036854775807);  slice_501 = slice_scatter_95 = None
    slice_scatter_97: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_500, slice_scatter_96, 1, 1, 9223372036854775807);  slice_500 = slice_scatter_96 = None
    slice_scatter_98: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_94, slice_scatter_97, 0, 0, 9223372036854775807);  slice_scatter_94 = slice_scatter_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_45: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_463, 1, 0);  slice_463 = None
    slice_512: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
    slice_513: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_512, 2, -255, 9223372036854775807);  slice_512 = None
    slice_517: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_98, 0, 0, 9223372036854775807)
    select_47: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_517, 1, 0)
    slice_518: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_47, 1, 1, 256)
    slice_519: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_518, 2, 1, 256)
    copy_27: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_519, slice_513);  slice_519 = slice_513 = None
    slice_scatter_99: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_518, copy_27, 2, 1, 256);  slice_518 = copy_27 = None
    slice_scatter_100: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_47, slice_scatter_99, 1, 1, 256);  select_47 = slice_scatter_99 = None
    select_scatter_9: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_517, slice_scatter_100, 1, 0);  slice_517 = slice_scatter_100 = None
    slice_scatter_101: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_98, select_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_98 = select_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_175: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_101, [1, 12, 1024, 513]);  slice_scatter_101 = None
    permute_145: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    slice_531: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_145, 0, 0, 9223372036854775807)
    slice_532: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_531, 1, 0, 256)
    slice_533: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_532, 2, 0, 9223372036854775807)
    slice_534: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_533, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_17: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_534)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_28: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_534, where_17);  slice_534 = where_17 = None
    slice_scatter_102: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_533, copy_28, 3, 0, 257);  slice_533 = copy_28 = None
    slice_scatter_103: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_532, slice_scatter_102, 2, 0, 9223372036854775807);  slice_532 = slice_scatter_102 = None
    slice_scatter_104: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_531, slice_scatter_103, 1, 0, 256);  slice_531 = slice_scatter_103 = None
    slice_scatter_105: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_145, slice_scatter_104, 0, 0, 9223372036854775807);  permute_145 = slice_scatter_104 = None
    permute_148: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_105, [0, 2, 1, 3]);  slice_scatter_105 = None
    view_178: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_148, [12, 4, 256, 513]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_180: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_178, [1, 12, 1024, 513]);  view_178 = None
    permute_150: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    slice_554: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_150, 0, 0, 9223372036854775807)
    slice_555: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_554, 1, -256, 9223372036854775807)
    slice_556: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_555, 2, 0, 9223372036854775807)
    slice_557: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_556, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_18: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_557)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_29: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_557, where_18);  slice_557 = where_18 = None
    slice_scatter_106: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_556, copy_29, 3, -257, 9223372036854775807);  slice_556 = copy_29 = None
    slice_scatter_107: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_555, slice_scatter_106, 2, 0, 9223372036854775807);  slice_555 = slice_scatter_106 = None
    slice_scatter_108: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_554, slice_scatter_107, 1, -256, 9223372036854775807);  slice_554 = slice_scatter_107 = None
    slice_scatter_109: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_150, slice_scatter_108, 0, 0, 9223372036854775807);  permute_150 = slice_scatter_108 = None
    permute_153: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None
    view_183: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_153, [12, 4, 256, 513]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_203: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_183, [1, 12, 1024, 513]);  view_183 = None
    permute_171: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    add_24: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_171, permute_46);  permute_171 = None
    permute_173: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_24, [0, 2, 1, 3]);  add_24 = None
    view_206: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_173, [12, 4, 256, 513]);  permute_173 = None
    view_207: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_206, [1, 12, 1024, 513]);  view_206 = None
    permute_174: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_12: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    amax_2: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_12, [-1], True)
    sub_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_12, amax_2);  clone_12 = amax_2 = None
    exp_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_3: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_27: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_23: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_6 = torch.ops.aten.native_dropout.default(where_23, 0.1, True);  where_23 = None
    getitem_20: "f32[1, 1024, 12, 513]" = native_dropout_6[0]
    getitem_21: "b8[1, 1024, 12, 513]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_208: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_155, [1024, 1, 12, 64]);  view_155 = None
    permute_175: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_208, [1, 0, 2, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_176: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
    view_209: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_176, [12, 4, 256, 513]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_177: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_175, [0, 2, 1, 3]);  permute_175 = None
    view_210: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_177, [12, 1024, 64]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_10: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_210, [0, 0, 256, 256], -1.0);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_17: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_11: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_209, [0, 257], 0.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_211: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_11, [12, 4, -1]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_687: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_211, 0, 0, 9223372036854775807);  view_211 = None
    slice_688: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_687, 1, 0, 9223372036854775807);  slice_687 = None
    slice_689: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_688, 2, 0, -256);  slice_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_212: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_689, [12, 4, 256, 769]);  slice_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_690: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_212, 0, 0, 9223372036854775807);  view_212 = None
    slice_691: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_690, 1, 0, 9223372036854775807);  slice_690 = None
    slice_692: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_691, 2, 0, 9223372036854775807);  slice_691 = None
    slice_693: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_692, 3, 0, -1);  slice_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_55: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_693, 4);  slice_693 = None
    permute_178: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_55, [0, 1, 2, 4, 3]);  unsqueeze_55 = None
    unsqueeze_56: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
    permute_179: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_56, [0, 1, 4, 3, 2]);  unsqueeze_56 = None
    permute_180: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_178, [0, 1, 2, 4, 3]);  permute_178 = None
    view_213: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_180, [48, 256, 768]);  permute_180 = None
    permute_181: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_179, [0, 1, 4, 3, 2]);  permute_179 = None
    clone_13: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_214: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_13, [48, 768, 64]);  clone_13 = None
    bmm_5: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_213, view_214)
    view_215: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_5, [12, 4, 256, 1, 64]);  bmm_5 = None
    permute_182: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_215, [0, 1, 2, 4, 3]);  view_215 = None
    view_216: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_182, [12, 4, 256, 64]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_217: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_216, [1, 12, 1024, 64]);  view_216 = None
    permute_183: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_184: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_183, [1, 0, 2, 3]);  permute_183 = None
    clone_14: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_218: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_14, [1024, 1, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_185: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_218, [1, 0, 2]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[1024, 768]" = torch.ops.aten.view.default(permute_185, [1024, 768]);  permute_185 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_40, view_219, permute_186);  primals_40 = None
    view_220: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_7 = torch.ops.aten.native_dropout.default(view_220, 0.1, True);  view_220 = None
    getitem_22: "f32[1, 1024, 768]" = native_dropout_7[0]
    getitem_23: "b8[1, 1024, 768]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_21);  getitem_22 = add_21 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_22: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_26, getitem_25);  add_26 = getitem_25 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_41)
    add_28: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_42);  mul_18 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_221: "f32[1024, 768]" = torch.ops.aten.view.default(add_28, [1024, 768])
    permute_187: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_16: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_44, view_221, permute_187);  primals_44 = None
    view_222: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, 0.5)
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
    erf_2: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_29: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_29);  mul_19 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_223: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_21, [1024, 3072]);  mul_21 = None
    permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_46, view_223, permute_188);  primals_46 = None
    view_224: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_224, 0.1, True);  view_224 = None
    getitem_26: "f32[1, 1024, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 1024, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_30: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_28);  getitem_26 = add_28 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_23: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_29);  add_30 = getitem_29 = None
    mul_22: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = None
    mul_23: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_47)
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_48);  mul_23 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_189: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_32, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_225: "f32[1024, 768]" = torch.ops.aten.view.default(permute_189, [1024, 768]);  permute_189 = None
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_18: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_50, view_225, permute_190);  primals_50 = None
    view_226: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_18, [1024, 1, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_52, view_225, permute_191);  primals_52 = None
    view_228: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_19, [1024, 1, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_20: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_54, view_225, permute_192);  primals_54 = None
    view_230: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_20, [1024, 1, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_30: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_226, 8.0);  view_226 = None
    view_231: "f32[1024, 768]" = torch.ops.aten.view.default(div_30, [1024, 768]);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_234: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_228, [1024, 1, 12, 64]);  view_228 = None
    permute_194: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_234, [1, 0, 2, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_196: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_194, [0, 2, 1, 3]);  permute_194 = None
    view_236: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_196, [12, 1024, 64]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_238: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_236, [12, 2, 512, 64]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_19: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_238, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_58: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
    permute_198: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 4, 2, 3]);  unsqueeze_58 = None
    view_239: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_231, [1024, 1, 768]);  view_231 = None
    view_240: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_239, [1024, 1, 12, 64]);  view_239 = None
    permute_200: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_240, [1, 0, 2, 3]);  view_240 = None
    permute_201: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_200, [0, 2, 1, 3]);  permute_200 = None
    view_241: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_201, [12, 1024, 64]);  permute_201 = None
    view_242: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_241, [12, 2, 512, 64]);  view_241 = None
    as_strided_20: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_242, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_242 = None
    unsqueeze_59: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
    permute_202: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_59, [0, 1, 2, 4, 3]);  unsqueeze_59 = None
    permute_203: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_202, [0, 1, 2, 4, 3]);  permute_202 = None
    clone_15: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_243: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_15, [36, 512, 64]);  clone_15 = None
    permute_204: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_198, [0, 1, 4, 3, 2]);  permute_198 = None
    clone_16: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_244: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_16, [36, 64, 512]);  clone_16 = None
    bmm_6: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_243, view_244)
    view_245: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, [12, 3, 512, 1, 512]);  bmm_6 = None
    permute_205: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_245, [0, 1, 2, 4, 3]);  view_245 = None
    view_246: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_205, [12, 3, 512, 512]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_12: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_246, [0, 0, 0, 1], 0.0);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_247: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_12, [12, 3, 512, 513]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_694: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_247, 0, 0, 9223372036854775807);  view_247 = None
    slice_695: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_694, 1, 0, 9223372036854775807)
    slice_696: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_695, 2, 0, 256)
    slice_697: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_696, 3, 0, 257);  slice_696 = None
    copy_36: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_697);  slice_697 = None
    slice_scatter_132: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_36, 3, 256, 9223372036854775807);  copy_36 = None
    slice_scatter_133: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_132, 2, 0, 9223372036854775807);  slice_scatter_132 = None
    slice_scatter_134: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_133, 1, 0, -1);  slice_scatter_133 = None
    slice_scatter_135: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_134, 0, 0, 9223372036854775807);  slice_scatter_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_60: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_694, 1, -1)
    slice_710: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_60, 1, 256, 9223372036854775807);  select_60 = None
    slice_711: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_710, 2, 0, 257);  slice_710 = None
    slice_715: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_135, 0, 0, 9223372036854775807)
    select_62: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_715, 1, -1)
    slice_716: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_62, 1, 0, 9223372036854775807)
    slice_717: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_716, 2, 256, 9223372036854775807)
    copy_37: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_717, slice_711);  slice_717 = slice_711 = None
    slice_scatter_136: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_716, copy_37, 2, 256, 9223372036854775807);  slice_716 = copy_37 = None
    slice_scatter_137: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_62, slice_scatter_136, 1, 0, 9223372036854775807);  select_62 = slice_scatter_136 = None
    select_scatter_12: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_715, slice_scatter_137, 1, -1);  slice_715 = slice_scatter_137 = None
    slice_scatter_138: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_135, select_scatter_12, 0, 0, 9223372036854775807);  slice_scatter_135 = select_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_725: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_695, 2, -257, -1);  slice_695 = None
    slice_726: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_725, 3, 257, 9223372036854775807);  slice_725 = None
    slice_731: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_138, 0, 0, 9223372036854775807)
    slice_732: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_731, 1, 1, 9223372036854775807)
    slice_733: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_732, 2, 0, 9223372036854775807)
    slice_734: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_733, 3, 0, 256)
    copy_38: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_734, slice_726);  slice_734 = slice_726 = None
    slice_scatter_139: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_733, copy_38, 3, 0, 256);  slice_733 = copy_38 = None
    slice_scatter_140: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_732, slice_scatter_139, 2, 0, 9223372036854775807);  slice_732 = slice_scatter_139 = None
    slice_scatter_141: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_731, slice_scatter_140, 1, 1, 9223372036854775807);  slice_731 = slice_scatter_140 = None
    slice_scatter_142: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_138, slice_scatter_141, 0, 0, 9223372036854775807);  slice_scatter_138 = slice_scatter_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_65: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_694, 1, 0);  slice_694 = None
    slice_743: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_65, 1, 0, 255);  select_65 = None
    slice_744: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_743, 2, -255, 9223372036854775807);  slice_743 = None
    slice_748: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_142, 0, 0, 9223372036854775807)
    select_67: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_748, 1, 0)
    slice_749: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_67, 1, 1, 256)
    slice_750: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_749, 2, 1, 256)
    copy_39: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_750, slice_744);  slice_750 = slice_744 = None
    slice_scatter_143: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_749, copy_39, 2, 1, 256);  slice_749 = copy_39 = None
    slice_scatter_144: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_67, slice_scatter_143, 1, 1, 256);  select_67 = slice_scatter_143 = None
    select_scatter_13: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_748, slice_scatter_144, 1, 0);  slice_748 = slice_scatter_144 = None
    slice_scatter_145: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_142, select_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_142 = select_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_250: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_145, [1, 12, 1024, 513]);  slice_scatter_145 = None
    permute_208: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    slice_762: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_208, 0, 0, 9223372036854775807)
    slice_763: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_762, 1, 0, 256)
    slice_764: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_763, 2, 0, 9223372036854775807)
    slice_765: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_764, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_25: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_765)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_40: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_765, where_25);  slice_765 = where_25 = None
    slice_scatter_146: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_764, copy_40, 3, 0, 257);  slice_764 = copy_40 = None
    slice_scatter_147: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_763, slice_scatter_146, 2, 0, 9223372036854775807);  slice_763 = slice_scatter_146 = None
    slice_scatter_148: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_762, slice_scatter_147, 1, 0, 256);  slice_762 = slice_scatter_147 = None
    slice_scatter_149: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_208, slice_scatter_148, 0, 0, 9223372036854775807);  permute_208 = slice_scatter_148 = None
    permute_211: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_149, [0, 2, 1, 3]);  slice_scatter_149 = None
    view_253: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_211, [12, 4, 256, 513]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_255: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_253, [1, 12, 1024, 513]);  view_253 = None
    permute_213: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    slice_785: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_213, 0, 0, 9223372036854775807)
    slice_786: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_785, 1, -256, 9223372036854775807)
    slice_787: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_786, 2, 0, 9223372036854775807)
    slice_788: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_787, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_26: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_788)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_41: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_788, where_26);  slice_788 = where_26 = None
    slice_scatter_150: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_787, copy_41, 3, -257, 9223372036854775807);  slice_787 = copy_41 = None
    slice_scatter_151: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_786, slice_scatter_150, 2, 0, 9223372036854775807);  slice_786 = slice_scatter_150 = None
    slice_scatter_152: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_785, slice_scatter_151, 1, -256, 9223372036854775807);  slice_785 = slice_scatter_151 = None
    slice_scatter_153: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_213, slice_scatter_152, 0, 0, 9223372036854775807);  permute_213 = slice_scatter_152 = None
    permute_216: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_153, [0, 2, 1, 3]);  slice_scatter_153 = None
    view_258: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_216, [12, 4, 256, 513]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_278: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_258, [1, 12, 1024, 513]);  view_258 = None
    permute_234: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    add_35: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_234, permute_46);  permute_234 = None
    permute_236: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
    view_281: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_236, [12, 4, 256, 513]);  permute_236 = None
    view_282: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_281, [1, 12, 1024, 513]);  view_281 = None
    permute_237: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    amax_3: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_17, [-1], True)
    sub_28: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_17, amax_3);  clone_17 = amax_3 = None
    exp_3: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_4: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_37: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_31: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_37);  div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_9 = torch.ops.aten.native_dropout.default(where_31, 0.1, True);  where_31 = None
    getitem_30: "f32[1, 1024, 12, 513]" = native_dropout_9[0]
    getitem_31: "b8[1, 1024, 12, 513]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_283: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_230, [1024, 1, 12, 64]);  view_230 = None
    permute_238: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_283, [1, 0, 2, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_239: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3]);  getitem_30 = None
    view_284: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_239, [12, 4, 256, 513]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_240: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_238, [0, 2, 1, 3]);  permute_238 = None
    view_285: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_240, [12, 1024, 64]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_14: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_285, [0, 0, 256, 256], -1.0);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_23: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_15: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_284, [0, 257], 0.0);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_286: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_15, [12, 4, -1]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_918: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_286, 0, 0, 9223372036854775807);  view_286 = None
    slice_919: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_918, 1, 0, 9223372036854775807);  slice_918 = None
    slice_920: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_919, 2, 0, -256);  slice_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_287: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_920, [12, 4, 256, 769]);  slice_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_921: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_287, 0, 0, 9223372036854775807);  view_287 = None
    slice_922: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_921, 1, 0, 9223372036854775807);  slice_921 = None
    slice_923: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_922, 2, 0, 9223372036854775807);  slice_922 = None
    slice_924: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_923, 3, 0, -1);  slice_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_74: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_924, 4);  slice_924 = None
    permute_241: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None
    unsqueeze_75: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
    permute_242: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
    permute_243: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_241, [0, 1, 2, 4, 3]);  permute_241 = None
    view_288: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_243, [48, 256, 768]);  permute_243 = None
    permute_244: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_242, [0, 1, 4, 3, 2]);  permute_242 = None
    clone_18: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_289: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_18, [48, 768, 64]);  clone_18 = None
    bmm_7: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_288, view_289)
    view_290: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_7, [12, 4, 256, 1, 64]);  bmm_7 = None
    permute_245: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_290, [0, 1, 2, 4, 3]);  view_290 = None
    view_291: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_245, [12, 4, 256, 64]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_292: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_291, [1, 12, 1024, 64]);  view_291 = None
    permute_246: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_247: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_246, [1, 0, 2, 3]);  permute_246 = None
    clone_19: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_293: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_19, [1024, 1, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_248: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_293, [1, 0, 2]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_294: "f32[1024, 768]" = torch.ops.aten.view.default(permute_248, [1024, 768]);  permute_248 = None
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_56, view_294, permute_249);  primals_56 = None
    view_295: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_10 = torch.ops.aten.native_dropout.default(view_295, 0.1, True);  view_295 = None
    getitem_32: "f32[1, 1024, 768]" = native_dropout_10[0]
    getitem_33: "b8[1, 1024, 768]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_32, add_32);  getitem_32 = add_32 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_38: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_35);  add_37 = getitem_35 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_57)
    add_39: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_58);  mul_26 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_296: "f32[1024, 768]" = torch.ops.aten.view.default(add_39, [1024, 768])
    permute_250: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_60, view_296, permute_250);  primals_60 = None
    view_297: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
    erf_3: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_40: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_29: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_40);  mul_27 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_298: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_29, [1024, 3072]);  mul_29 = None
    permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_62, view_298, permute_251);  primals_62 = None
    view_299: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_299, 0.1, True);  view_299 = None
    getitem_36: "f32[1, 1024, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 1024, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_39);  getitem_36 = add_39 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_42: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_31: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_39);  add_41 = getitem_39 = None
    mul_30: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = None
    mul_31: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_63)
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_64);  mul_31 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_252: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_43, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_300: "f32[1024, 768]" = torch.ops.aten.view.default(permute_252, [1024, 768]);  permute_252 = None
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_24: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_66, view_300, permute_253);  primals_66 = None
    view_301: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_24, [1024, 1, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_68, view_300, permute_254);  primals_68 = None
    view_303: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_25, [1024, 1, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_26: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_70, view_300, permute_255);  primals_70 = None
    view_305: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_26, [1024, 1, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_40: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_301, 8.0);  view_301 = None
    view_306: "f32[1024, 768]" = torch.ops.aten.view.default(div_40, [1024, 768]);  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_309: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_303, [1024, 1, 12, 64]);  view_303 = None
    permute_257: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_309, [1, 0, 2, 3]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_259: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None
    view_311: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_259, [12, 1024, 64]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_313: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_311, [12, 2, 512, 64]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_25: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_313, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_77: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
    permute_261: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_77, [0, 1, 4, 2, 3]);  unsqueeze_77 = None
    view_314: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_306, [1024, 1, 768]);  view_306 = None
    view_315: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_314, [1024, 1, 12, 64]);  view_314 = None
    permute_263: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_315, [1, 0, 2, 3]);  view_315 = None
    permute_264: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_263, [0, 2, 1, 3]);  permute_263 = None
    view_316: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_264, [12, 1024, 64]);  permute_264 = None
    view_317: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_316, [12, 2, 512, 64]);  view_316 = None
    as_strided_26: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_317, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_317 = None
    unsqueeze_78: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
    permute_265: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_78, [0, 1, 2, 4, 3]);  unsqueeze_78 = None
    permute_266: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_265, [0, 1, 2, 4, 3]);  permute_265 = None
    clone_20: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_318: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_20, [36, 512, 64]);  clone_20 = None
    permute_267: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_261, [0, 1, 4, 3, 2]);  permute_261 = None
    clone_21: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_319: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_21, [36, 64, 512]);  clone_21 = None
    bmm_8: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_318, view_319)
    view_320: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_8, [12, 3, 512, 1, 512]);  bmm_8 = None
    permute_268: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_320, [0, 1, 2, 4, 3]);  view_320 = None
    view_321: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_268, [12, 3, 512, 512]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_16: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_321, [0, 0, 0, 1], 0.0);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_322: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_16, [12, 3, 512, 513]);  constant_pad_nd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_925: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_322, 0, 0, 9223372036854775807);  view_322 = None
    slice_926: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_925, 1, 0, 9223372036854775807)
    slice_927: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_926, 2, 0, 256)
    slice_928: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_927, 3, 0, 257);  slice_927 = None
    copy_48: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_928);  slice_928 = None
    slice_scatter_176: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_48, 3, 256, 9223372036854775807);  copy_48 = None
    slice_scatter_177: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_176, 2, 0, 9223372036854775807);  slice_scatter_176 = None
    slice_scatter_178: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_177, 1, 0, -1);  slice_scatter_177 = None
    slice_scatter_179: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_178, 0, 0, 9223372036854775807);  slice_scatter_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_80: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_925, 1, -1)
    slice_941: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_80, 1, 256, 9223372036854775807);  select_80 = None
    slice_942: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_941, 2, 0, 257);  slice_941 = None
    slice_946: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_179, 0, 0, 9223372036854775807)
    select_82: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_946, 1, -1)
    slice_947: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_82, 1, 0, 9223372036854775807)
    slice_948: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_947, 2, 256, 9223372036854775807)
    copy_49: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_948, slice_942);  slice_948 = slice_942 = None
    slice_scatter_180: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_947, copy_49, 2, 256, 9223372036854775807);  slice_947 = copy_49 = None
    slice_scatter_181: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_82, slice_scatter_180, 1, 0, 9223372036854775807);  select_82 = slice_scatter_180 = None
    select_scatter_16: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_946, slice_scatter_181, 1, -1);  slice_946 = slice_scatter_181 = None
    slice_scatter_182: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_179, select_scatter_16, 0, 0, 9223372036854775807);  slice_scatter_179 = select_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_956: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_926, 2, -257, -1);  slice_926 = None
    slice_957: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_956, 3, 257, 9223372036854775807);  slice_956 = None
    slice_962: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_182, 0, 0, 9223372036854775807)
    slice_963: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_962, 1, 1, 9223372036854775807)
    slice_964: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_963, 2, 0, 9223372036854775807)
    slice_965: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_964, 3, 0, 256)
    copy_50: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_965, slice_957);  slice_965 = slice_957 = None
    slice_scatter_183: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_964, copy_50, 3, 0, 256);  slice_964 = copy_50 = None
    slice_scatter_184: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_963, slice_scatter_183, 2, 0, 9223372036854775807);  slice_963 = slice_scatter_183 = None
    slice_scatter_185: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_962, slice_scatter_184, 1, 1, 9223372036854775807);  slice_962 = slice_scatter_184 = None
    slice_scatter_186: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_182, slice_scatter_185, 0, 0, 9223372036854775807);  slice_scatter_182 = slice_scatter_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_85: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_925, 1, 0);  slice_925 = None
    slice_974: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_85, 1, 0, 255);  select_85 = None
    slice_975: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_974, 2, -255, 9223372036854775807);  slice_974 = None
    slice_979: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_186, 0, 0, 9223372036854775807)
    select_87: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_979, 1, 0)
    slice_980: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_87, 1, 1, 256)
    slice_981: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_980, 2, 1, 256)
    copy_51: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_981, slice_975);  slice_981 = slice_975 = None
    slice_scatter_187: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_980, copy_51, 2, 1, 256);  slice_980 = copy_51 = None
    slice_scatter_188: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_87, slice_scatter_187, 1, 1, 256);  select_87 = slice_scatter_187 = None
    select_scatter_17: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_979, slice_scatter_188, 1, 0);  slice_979 = slice_scatter_188 = None
    slice_scatter_189: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_186, select_scatter_17, 0, 0, 9223372036854775807);  slice_scatter_186 = select_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_325: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_189, [1, 12, 1024, 513]);  slice_scatter_189 = None
    permute_271: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    slice_993: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_271, 0, 0, 9223372036854775807)
    slice_994: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_993, 1, 0, 256)
    slice_995: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_994, 2, 0, 9223372036854775807)
    slice_996: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_995, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_33: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_996)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_52: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_996, where_33);  slice_996 = where_33 = None
    slice_scatter_190: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_995, copy_52, 3, 0, 257);  slice_995 = copy_52 = None
    slice_scatter_191: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_994, slice_scatter_190, 2, 0, 9223372036854775807);  slice_994 = slice_scatter_190 = None
    slice_scatter_192: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_993, slice_scatter_191, 1, 0, 256);  slice_993 = slice_scatter_191 = None
    slice_scatter_193: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_271, slice_scatter_192, 0, 0, 9223372036854775807);  permute_271 = slice_scatter_192 = None
    permute_274: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_193, [0, 2, 1, 3]);  slice_scatter_193 = None
    view_328: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_274, [12, 4, 256, 513]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_330: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_328, [1, 12, 1024, 513]);  view_328 = None
    permute_276: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    slice_1016: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_276, 0, 0, 9223372036854775807)
    slice_1017: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1016, 1, -256, 9223372036854775807)
    slice_1018: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1017, 2, 0, 9223372036854775807)
    slice_1019: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1018, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_34: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_1019)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_53: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1019, where_34);  slice_1019 = where_34 = None
    slice_scatter_194: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1018, copy_53, 3, -257, 9223372036854775807);  slice_1018 = copy_53 = None
    slice_scatter_195: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1017, slice_scatter_194, 2, 0, 9223372036854775807);  slice_1017 = slice_scatter_194 = None
    slice_scatter_196: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1016, slice_scatter_195, 1, -256, 9223372036854775807);  slice_1016 = slice_scatter_195 = None
    slice_scatter_197: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_276, slice_scatter_196, 0, 0, 9223372036854775807);  permute_276 = slice_scatter_196 = None
    permute_279: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_197, [0, 2, 1, 3]);  slice_scatter_197 = None
    view_333: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_279, [12, 4, 256, 513]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_353: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_333, [1, 12, 1024, 513]);  view_333 = None
    permute_297: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    add_46: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_297, permute_46);  permute_297 = None
    permute_299: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_46, [0, 2, 1, 3]);  add_46 = None
    view_356: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_299, [12, 4, 256, 513]);  permute_299 = None
    view_357: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_356, [1, 12, 1024, 513]);  view_356 = None
    permute_300: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_22: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    amax_4: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_22, [-1], True)
    sub_36: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_22, amax_4);  clone_22 = amax_4 = None
    exp_4: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_5: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_47: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_39: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_12 = torch.ops.aten.native_dropout.default(where_39, 0.1, True);  where_39 = None
    getitem_40: "f32[1, 1024, 12, 513]" = native_dropout_12[0]
    getitem_41: "b8[1, 1024, 12, 513]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_358: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_305, [1024, 1, 12, 64]);  view_305 = None
    permute_301: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_358, [1, 0, 2, 3]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_302: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_40, [0, 2, 1, 3]);  getitem_40 = None
    view_359: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_302, [12, 4, 256, 513]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_303: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
    view_360: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_303, [12, 1024, 64]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_18: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_360, [0, 0, 256, 256], -1.0);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_29: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_19: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_359, [0, 257], 0.0);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_361: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_19, [12, 4, -1]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1149: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_361, 0, 0, 9223372036854775807);  view_361 = None
    slice_1150: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1149, 1, 0, 9223372036854775807);  slice_1149 = None
    slice_1151: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1150, 2, 0, -256);  slice_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_362: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1151, [12, 4, 256, 769]);  slice_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1152: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_362, 0, 0, 9223372036854775807);  view_362 = None
    slice_1153: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1152, 1, 0, 9223372036854775807);  slice_1152 = None
    slice_1154: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1153, 2, 0, 9223372036854775807);  slice_1153 = None
    slice_1155: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1154, 3, 0, -1);  slice_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_93: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1155, 4);  slice_1155 = None
    permute_304: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_93, [0, 1, 2, 4, 3]);  unsqueeze_93 = None
    unsqueeze_94: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
    permute_305: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_94, [0, 1, 4, 3, 2]);  unsqueeze_94 = None
    permute_306: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_304, [0, 1, 2, 4, 3]);  permute_304 = None
    view_363: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_306, [48, 256, 768]);  permute_306 = None
    permute_307: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_305, [0, 1, 4, 3, 2]);  permute_305 = None
    clone_23: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    view_364: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_23, [48, 768, 64]);  clone_23 = None
    bmm_9: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_363, view_364)
    view_365: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_9, [12, 4, 256, 1, 64]);  bmm_9 = None
    permute_308: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_365, [0, 1, 2, 4, 3]);  view_365 = None
    view_366: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_308, [12, 4, 256, 64]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_367: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_366, [1, 12, 1024, 64]);  view_366 = None
    permute_309: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_310: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_309, [1, 0, 2, 3]);  permute_309 = None
    clone_24: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
    view_368: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_24, [1024, 1, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_311: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_368, [1, 0, 2]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_369: "f32[1024, 768]" = torch.ops.aten.view.default(permute_311, [1024, 768]);  permute_311 = None
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_72, view_369, permute_312);  primals_72 = None
    view_370: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_13 = torch.ops.aten.native_dropout.default(view_370, 0.1, True);  view_370 = None
    getitem_42: "f32[1, 1024, 768]" = native_dropout_13[0]
    getitem_43: "b8[1, 1024, 768]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_42, add_43);  getitem_42 = add_43 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_38: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_45);  add_48 = getitem_45 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_73)
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_74);  mul_34 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_371: "f32[1024, 768]" = torch.ops.aten.view.default(add_50, [1024, 768])
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_28: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_76, view_371, permute_313);  primals_76 = None
    view_372: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, 0.5)
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
    erf_4: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_51: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_37: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_51);  mul_35 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_373: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_37, [1024, 3072]);  mul_37 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_78, view_373, permute_314);  primals_78 = None
    view_374: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_374, 0.1, True);  view_374 = None
    getitem_46: "f32[1, 1024, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 1024, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_52: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_50);  getitem_46 = add_50 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_39: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_49);  add_52 = getitem_49 = None
    mul_38: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = None
    mul_39: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_79)
    add_54: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_80);  mul_39 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_315: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_54, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_375: "f32[1024, 768]" = torch.ops.aten.view.default(permute_315, [1024, 768]);  permute_315 = None
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_30: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_82, view_375, permute_316);  primals_82 = None
    view_376: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_30, [1024, 1, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_84, view_375, permute_317);  primals_84 = None
    view_378: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_31, [1024, 1, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_32: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_86, view_375, permute_318);  primals_86 = None
    view_380: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_32, [1024, 1, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_50: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_376, 8.0);  view_376 = None
    view_381: "f32[1024, 768]" = torch.ops.aten.view.default(div_50, [1024, 768]);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_384: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_378, [1024, 1, 12, 64]);  view_378 = None
    permute_320: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_384, [1, 0, 2, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_322: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
    view_386: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_322, [12, 1024, 64]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_388: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_386, [12, 2, 512, 64]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_31: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_388, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_96: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
    permute_324: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
    view_389: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_381, [1024, 1, 768]);  view_381 = None
    view_390: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_389, [1024, 1, 12, 64]);  view_389 = None
    permute_326: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_390, [1, 0, 2, 3]);  view_390 = None
    permute_327: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
    view_391: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_327, [12, 1024, 64]);  permute_327 = None
    view_392: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_391, [12, 2, 512, 64]);  view_391 = None
    as_strided_32: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_392, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_392 = None
    unsqueeze_97: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
    permute_328: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_97, [0, 1, 2, 4, 3]);  unsqueeze_97 = None
    permute_329: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_328, [0, 1, 2, 4, 3]);  permute_328 = None
    clone_25: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_393: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_25, [36, 512, 64]);  clone_25 = None
    permute_330: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_324, [0, 1, 4, 3, 2]);  permute_324 = None
    clone_26: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
    view_394: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_26, [36, 64, 512]);  clone_26 = None
    bmm_10: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_393, view_394)
    view_395: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_10, [12, 3, 512, 1, 512]);  bmm_10 = None
    permute_331: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_395, [0, 1, 2, 4, 3]);  view_395 = None
    view_396: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_331, [12, 3, 512, 512]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_20: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_396, [0, 0, 0, 1], 0.0);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_397: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_20, [12, 3, 512, 513]);  constant_pad_nd_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1156: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_397, 0, 0, 9223372036854775807);  view_397 = None
    slice_1157: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1156, 1, 0, 9223372036854775807)
    slice_1158: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1157, 2, 0, 256)
    slice_1159: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1158, 3, 0, 257);  slice_1158 = None
    copy_60: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_1159);  slice_1159 = None
    slice_scatter_220: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_60, 3, 256, 9223372036854775807);  copy_60 = None
    slice_scatter_221: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_220, 2, 0, 9223372036854775807);  slice_scatter_220 = None
    slice_scatter_222: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_221, 1, 0, -1);  slice_scatter_221 = None
    slice_scatter_223: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_222, 0, 0, 9223372036854775807);  slice_scatter_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_100: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1156, 1, -1)
    slice_1172: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_100, 1, 256, 9223372036854775807);  select_100 = None
    slice_1173: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1172, 2, 0, 257);  slice_1172 = None
    slice_1177: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_223, 0, 0, 9223372036854775807)
    select_102: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1177, 1, -1)
    slice_1178: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_102, 1, 0, 9223372036854775807)
    slice_1179: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1178, 2, 256, 9223372036854775807)
    copy_61: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1179, slice_1173);  slice_1179 = slice_1173 = None
    slice_scatter_224: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1178, copy_61, 2, 256, 9223372036854775807);  slice_1178 = copy_61 = None
    slice_scatter_225: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_102, slice_scatter_224, 1, 0, 9223372036854775807);  select_102 = slice_scatter_224 = None
    select_scatter_20: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1177, slice_scatter_225, 1, -1);  slice_1177 = slice_scatter_225 = None
    slice_scatter_226: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_223, select_scatter_20, 0, 0, 9223372036854775807);  slice_scatter_223 = select_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1187: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1157, 2, -257, -1);  slice_1157 = None
    slice_1188: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1187, 3, 257, 9223372036854775807);  slice_1187 = None
    slice_1193: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_226, 0, 0, 9223372036854775807)
    slice_1194: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1193, 1, 1, 9223372036854775807)
    slice_1195: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1194, 2, 0, 9223372036854775807)
    slice_1196: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1195, 3, 0, 256)
    copy_62: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1196, slice_1188);  slice_1196 = slice_1188 = None
    slice_scatter_227: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1195, copy_62, 3, 0, 256);  slice_1195 = copy_62 = None
    slice_scatter_228: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1194, slice_scatter_227, 2, 0, 9223372036854775807);  slice_1194 = slice_scatter_227 = None
    slice_scatter_229: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1193, slice_scatter_228, 1, 1, 9223372036854775807);  slice_1193 = slice_scatter_228 = None
    slice_scatter_230: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_226, slice_scatter_229, 0, 0, 9223372036854775807);  slice_scatter_226 = slice_scatter_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_105: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1156, 1, 0);  slice_1156 = None
    slice_1205: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_105, 1, 0, 255);  select_105 = None
    slice_1206: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1205, 2, -255, 9223372036854775807);  slice_1205 = None
    slice_1210: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_230, 0, 0, 9223372036854775807)
    select_107: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1210, 1, 0)
    slice_1211: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_107, 1, 1, 256)
    slice_1212: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1211, 2, 1, 256)
    copy_63: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1212, slice_1206);  slice_1212 = slice_1206 = None
    slice_scatter_231: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1211, copy_63, 2, 1, 256);  slice_1211 = copy_63 = None
    slice_scatter_232: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_107, slice_scatter_231, 1, 1, 256);  select_107 = slice_scatter_231 = None
    select_scatter_21: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1210, slice_scatter_232, 1, 0);  slice_1210 = slice_scatter_232 = None
    slice_scatter_233: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_230, select_scatter_21, 0, 0, 9223372036854775807);  slice_scatter_230 = select_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_400: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_233, [1, 12, 1024, 513]);  slice_scatter_233 = None
    permute_334: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    slice_1224: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_334, 0, 0, 9223372036854775807)
    slice_1225: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1224, 1, 0, 256)
    slice_1226: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1225, 2, 0, 9223372036854775807)
    slice_1227: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1226, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_41: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_1227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_64: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1227, where_41);  slice_1227 = where_41 = None
    slice_scatter_234: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1226, copy_64, 3, 0, 257);  slice_1226 = copy_64 = None
    slice_scatter_235: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1225, slice_scatter_234, 2, 0, 9223372036854775807);  slice_1225 = slice_scatter_234 = None
    slice_scatter_236: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1224, slice_scatter_235, 1, 0, 256);  slice_1224 = slice_scatter_235 = None
    slice_scatter_237: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_334, slice_scatter_236, 0, 0, 9223372036854775807);  permute_334 = slice_scatter_236 = None
    permute_337: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_237, [0, 2, 1, 3]);  slice_scatter_237 = None
    view_403: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_337, [12, 4, 256, 513]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_405: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_403, [1, 12, 1024, 513]);  view_403 = None
    permute_339: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    slice_1247: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_339, 0, 0, 9223372036854775807)
    slice_1248: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1247, 1, -256, 9223372036854775807)
    slice_1249: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1248, 2, 0, 9223372036854775807)
    slice_1250: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1249, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_42: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_1250)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_65: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1250, where_42);  slice_1250 = where_42 = None
    slice_scatter_238: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1249, copy_65, 3, -257, 9223372036854775807);  slice_1249 = copy_65 = None
    slice_scatter_239: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1248, slice_scatter_238, 2, 0, 9223372036854775807);  slice_1248 = slice_scatter_238 = None
    slice_scatter_240: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1247, slice_scatter_239, 1, -256, 9223372036854775807);  slice_1247 = slice_scatter_239 = None
    slice_scatter_241: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_339, slice_scatter_240, 0, 0, 9223372036854775807);  permute_339 = slice_scatter_240 = None
    permute_342: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_241, [0, 2, 1, 3]);  slice_scatter_241 = None
    view_408: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_342, [12, 4, 256, 513]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_428: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_408, [1, 12, 1024, 513]);  view_408 = None
    permute_360: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    add_57: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_360, permute_46);  permute_360 = None
    permute_362: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_57, [0, 2, 1, 3]);  add_57 = None
    view_431: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_362, [12, 4, 256, 513]);  permute_362 = None
    view_432: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_431, [1, 12, 1024, 513]);  view_431 = None
    permute_363: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_27: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    amax_5: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_27, [-1], True)
    sub_44: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_27, amax_5);  clone_27 = amax_5 = None
    exp_5: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_6: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_57: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_47: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_57);  div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_15 = torch.ops.aten.native_dropout.default(where_47, 0.1, True);  where_47 = None
    getitem_50: "f32[1, 1024, 12, 513]" = native_dropout_15[0]
    getitem_51: "b8[1, 1024, 12, 513]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_433: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_380, [1024, 1, 12, 64]);  view_380 = None
    permute_364: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_433, [1, 0, 2, 3]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_365: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
    view_434: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_365, [12, 4, 256, 513]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_366: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_364, [0, 2, 1, 3]);  permute_364 = None
    view_435: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_366, [12, 1024, 64]);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_22: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_435, [0, 0, 256, 256], -1.0);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_35: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_23: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_434, [0, 257], 0.0);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_436: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_23, [12, 4, -1]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1380: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_436, 0, 0, 9223372036854775807);  view_436 = None
    slice_1381: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1380, 1, 0, 9223372036854775807);  slice_1380 = None
    slice_1382: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1381, 2, 0, -256);  slice_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_437: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1382, [12, 4, 256, 769]);  slice_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1383: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_437, 0, 0, 9223372036854775807);  view_437 = None
    slice_1384: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1383, 1, 0, 9223372036854775807);  slice_1383 = None
    slice_1385: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1384, 2, 0, 9223372036854775807);  slice_1384 = None
    slice_1386: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1385, 3, 0, -1);  slice_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_112: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1386, 4);  slice_1386 = None
    permute_367: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 2, 4, 3]);  unsqueeze_112 = None
    unsqueeze_113: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
    permute_368: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_113, [0, 1, 4, 3, 2]);  unsqueeze_113 = None
    permute_369: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_367, [0, 1, 2, 4, 3]);  permute_367 = None
    view_438: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_369, [48, 256, 768]);  permute_369 = None
    permute_370: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_368, [0, 1, 4, 3, 2]);  permute_368 = None
    clone_28: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_439: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_28, [48, 768, 64]);  clone_28 = None
    bmm_11: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_438, view_439)
    view_440: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_11, [12, 4, 256, 1, 64]);  bmm_11 = None
    permute_371: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
    view_441: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_371, [12, 4, 256, 64]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_442: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_441, [1, 12, 1024, 64]);  view_441 = None
    permute_372: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_373: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_372, [1, 0, 2, 3]);  permute_372 = None
    clone_29: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
    view_443: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_29, [1024, 1, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_374: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_443, [1, 0, 2]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_444: "f32[1024, 768]" = torch.ops.aten.view.default(permute_374, [1024, 768]);  permute_374 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_88, view_444, permute_375);  primals_88 = None
    view_445: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_16 = torch.ops.aten.native_dropout.default(view_445, 0.1, True);  view_445 = None
    getitem_52: "f32[1, 1024, 768]" = native_dropout_16[0]
    getitem_53: "b8[1, 1024, 768]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_52, add_54);  getitem_52 = add_54 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_46: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  add_59 = getitem_55 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_89)
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_90);  mul_42 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_446: "f32[1024, 768]" = torch.ops.aten.view.default(add_61, [1024, 768])
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_92, view_446, permute_376);  primals_92 = None
    view_447: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_43: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, 0.7071067811865476);  view_447 = None
    erf_5: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_45: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_62);  mul_43 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_45, [1024, 3072]);  mul_45 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_94, view_448, permute_377);  primals_94 = None
    view_449: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_449, 0.1, True);  view_449 = None
    getitem_56: "f32[1, 1024, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 1024, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_61);  getitem_56 = add_61 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_64: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_47: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_59);  add_63 = getitem_59 = None
    mul_46: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = None
    mul_47: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, primals_95)
    add_65: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, primals_96);  mul_47 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_378: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_65, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_450: "f32[1024, 768]" = torch.ops.aten.view.default(permute_378, [1024, 768]);  permute_378 = None
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_36: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_98, view_450, permute_379);  primals_98 = None
    view_451: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_36, [1024, 1, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_380: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_100, view_450, permute_380);  primals_100 = None
    view_453: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_37, [1024, 1, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_38: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_102, view_450, permute_381);  primals_102 = None
    view_455: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_38, [1024, 1, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_60: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_451, 8.0);  view_451 = None
    view_456: "f32[1024, 768]" = torch.ops.aten.view.default(div_60, [1024, 768]);  div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_459: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_453, [1024, 1, 12, 64]);  view_453 = None
    permute_383: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_459, [1, 0, 2, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_385: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_383, [0, 2, 1, 3]);  permute_383 = None
    view_461: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_385, [12, 1024, 64]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_463: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_461, [12, 2, 512, 64]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_37: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_463, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_115: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
    permute_387: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_115, [0, 1, 4, 2, 3]);  unsqueeze_115 = None
    view_464: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_456, [1024, 1, 768]);  view_456 = None
    view_465: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_464, [1024, 1, 12, 64]);  view_464 = None
    permute_389: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_465, [1, 0, 2, 3]);  view_465 = None
    permute_390: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
    view_466: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_390, [12, 1024, 64]);  permute_390 = None
    view_467: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_466, [12, 2, 512, 64]);  view_466 = None
    as_strided_38: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_467, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_467 = None
    unsqueeze_116: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_38, 4);  as_strided_38 = None
    permute_391: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None
    permute_392: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_391, [0, 1, 2, 4, 3]);  permute_391 = None
    clone_30: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_468: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_30, [36, 512, 64]);  clone_30 = None
    permute_393: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_387, [0, 1, 4, 3, 2]);  permute_387 = None
    clone_31: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_469: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_31, [36, 64, 512]);  clone_31 = None
    bmm_12: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_468, view_469)
    view_470: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, [12, 3, 512, 1, 512]);  bmm_12 = None
    permute_394: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_470, [0, 1, 2, 4, 3]);  view_470 = None
    view_471: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_394, [12, 3, 512, 512]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_24: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_471, [0, 0, 0, 1], 0.0);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_472: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_24, [12, 3, 512, 513]);  constant_pad_nd_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1387: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_472, 0, 0, 9223372036854775807);  view_472 = None
    slice_1388: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1387, 1, 0, 9223372036854775807)
    slice_1389: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1388, 2, 0, 256)
    slice_1390: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1389, 3, 0, 257);  slice_1389 = None
    copy_72: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_1390);  slice_1390 = None
    slice_scatter_264: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_72, 3, 256, 9223372036854775807);  copy_72 = None
    slice_scatter_265: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_264, 2, 0, 9223372036854775807);  slice_scatter_264 = None
    slice_scatter_266: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_265, 1, 0, -1);  slice_scatter_265 = None
    slice_scatter_267: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_266, 0, 0, 9223372036854775807);  slice_scatter_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_120: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1387, 1, -1)
    slice_1403: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_120, 1, 256, 9223372036854775807);  select_120 = None
    slice_1404: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1403, 2, 0, 257);  slice_1403 = None
    slice_1408: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_267, 0, 0, 9223372036854775807)
    select_122: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1408, 1, -1)
    slice_1409: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_122, 1, 0, 9223372036854775807)
    slice_1410: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1409, 2, 256, 9223372036854775807)
    copy_73: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1410, slice_1404);  slice_1410 = slice_1404 = None
    slice_scatter_268: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1409, copy_73, 2, 256, 9223372036854775807);  slice_1409 = copy_73 = None
    slice_scatter_269: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_122, slice_scatter_268, 1, 0, 9223372036854775807);  select_122 = slice_scatter_268 = None
    select_scatter_24: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1408, slice_scatter_269, 1, -1);  slice_1408 = slice_scatter_269 = None
    slice_scatter_270: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_267, select_scatter_24, 0, 0, 9223372036854775807);  slice_scatter_267 = select_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1418: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1388, 2, -257, -1);  slice_1388 = None
    slice_1419: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1418, 3, 257, 9223372036854775807);  slice_1418 = None
    slice_1424: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_270, 0, 0, 9223372036854775807)
    slice_1425: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1424, 1, 1, 9223372036854775807)
    slice_1426: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1425, 2, 0, 9223372036854775807)
    slice_1427: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1426, 3, 0, 256)
    copy_74: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1427, slice_1419);  slice_1427 = slice_1419 = None
    slice_scatter_271: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1426, copy_74, 3, 0, 256);  slice_1426 = copy_74 = None
    slice_scatter_272: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1425, slice_scatter_271, 2, 0, 9223372036854775807);  slice_1425 = slice_scatter_271 = None
    slice_scatter_273: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1424, slice_scatter_272, 1, 1, 9223372036854775807);  slice_1424 = slice_scatter_272 = None
    slice_scatter_274: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_270, slice_scatter_273, 0, 0, 9223372036854775807);  slice_scatter_270 = slice_scatter_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_125: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1387, 1, 0);  slice_1387 = None
    slice_1436: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_125, 1, 0, 255);  select_125 = None
    slice_1437: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1436, 2, -255, 9223372036854775807);  slice_1436 = None
    slice_1441: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_274, 0, 0, 9223372036854775807)
    select_127: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1441, 1, 0)
    slice_1442: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256)
    slice_1443: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1442, 2, 1, 256)
    copy_75: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1443, slice_1437);  slice_1443 = slice_1437 = None
    slice_scatter_275: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1442, copy_75, 2, 1, 256);  slice_1442 = copy_75 = None
    slice_scatter_276: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_127, slice_scatter_275, 1, 1, 256);  select_127 = slice_scatter_275 = None
    select_scatter_25: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1441, slice_scatter_276, 1, 0);  slice_1441 = slice_scatter_276 = None
    slice_scatter_277: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_274, select_scatter_25, 0, 0, 9223372036854775807);  slice_scatter_274 = select_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_475: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_277, [1, 12, 1024, 513]);  slice_scatter_277 = None
    permute_397: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    slice_1455: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_397, 0, 0, 9223372036854775807)
    slice_1456: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1455, 1, 0, 256)
    slice_1457: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1456, 2, 0, 9223372036854775807)
    slice_1458: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1457, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_49: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_1458)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_76: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1458, where_49);  slice_1458 = where_49 = None
    slice_scatter_278: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1457, copy_76, 3, 0, 257);  slice_1457 = copy_76 = None
    slice_scatter_279: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1456, slice_scatter_278, 2, 0, 9223372036854775807);  slice_1456 = slice_scatter_278 = None
    slice_scatter_280: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1455, slice_scatter_279, 1, 0, 256);  slice_1455 = slice_scatter_279 = None
    slice_scatter_281: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_397, slice_scatter_280, 0, 0, 9223372036854775807);  permute_397 = slice_scatter_280 = None
    permute_400: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_281, [0, 2, 1, 3]);  slice_scatter_281 = None
    view_478: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_400, [12, 4, 256, 513]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_480: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_478, [1, 12, 1024, 513]);  view_478 = None
    permute_402: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    slice_1478: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_402, 0, 0, 9223372036854775807)
    slice_1479: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1478, 1, -256, 9223372036854775807)
    slice_1480: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1479, 2, 0, 9223372036854775807)
    slice_1481: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1480, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_50: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_1481)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_77: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1481, where_50);  slice_1481 = where_50 = None
    slice_scatter_282: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1480, copy_77, 3, -257, 9223372036854775807);  slice_1480 = copy_77 = None
    slice_scatter_283: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1479, slice_scatter_282, 2, 0, 9223372036854775807);  slice_1479 = slice_scatter_282 = None
    slice_scatter_284: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1478, slice_scatter_283, 1, -256, 9223372036854775807);  slice_1478 = slice_scatter_283 = None
    slice_scatter_285: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_402, slice_scatter_284, 0, 0, 9223372036854775807);  permute_402 = slice_scatter_284 = None
    permute_405: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_285, [0, 2, 1, 3]);  slice_scatter_285 = None
    view_483: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_405, [12, 4, 256, 513]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_503: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_483, [1, 12, 1024, 513]);  view_483 = None
    permute_423: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
    add_68: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_423, permute_46);  permute_423 = None
    permute_425: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_68, [0, 2, 1, 3]);  add_68 = None
    view_506: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_425, [12, 4, 256, 513]);  permute_425 = None
    view_507: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_506, [1, 12, 1024, 513]);  view_506 = None
    permute_426: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_32: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    amax_6: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_32, [-1], True)
    sub_52: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_32, amax_6);  clone_32 = amax_6 = None
    exp_6: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_7: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_67: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_55: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_67);  div_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_18 = torch.ops.aten.native_dropout.default(where_55, 0.1, True);  where_55 = None
    getitem_60: "f32[1, 1024, 12, 513]" = native_dropout_18[0]
    getitem_61: "b8[1, 1024, 12, 513]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_508: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_455, [1024, 1, 12, 64]);  view_455 = None
    permute_427: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_508, [1, 0, 2, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_428: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_509: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_428, [12, 4, 256, 513]);  permute_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_429: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_427, [0, 2, 1, 3]);  permute_427 = None
    view_510: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_429, [12, 1024, 64]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_26: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_510, [0, 0, 256, 256], -1.0);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_41: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_27: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_509, [0, 257], 0.0);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_511: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_27, [12, 4, -1]);  constant_pad_nd_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1611: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_511, 0, 0, 9223372036854775807);  view_511 = None
    slice_1612: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1611, 1, 0, 9223372036854775807);  slice_1611 = None
    slice_1613: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1612, 2, 0, -256);  slice_1612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_512: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1613, [12, 4, 256, 769]);  slice_1613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1614: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_512, 0, 0, 9223372036854775807);  view_512 = None
    slice_1615: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1614, 1, 0, 9223372036854775807);  slice_1614 = None
    slice_1616: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1615, 2, 0, 9223372036854775807);  slice_1615 = None
    slice_1617: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1616, 3, 0, -1);  slice_1616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_131: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1617, 4);  slice_1617 = None
    permute_430: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_131, [0, 1, 2, 4, 3]);  unsqueeze_131 = None
    unsqueeze_132: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
    permute_431: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_132, [0, 1, 4, 3, 2]);  unsqueeze_132 = None
    permute_432: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_430, [0, 1, 2, 4, 3]);  permute_430 = None
    view_513: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_432, [48, 256, 768]);  permute_432 = None
    permute_433: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_431, [0, 1, 4, 3, 2]);  permute_431 = None
    clone_33: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_514: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_33, [48, 768, 64]);  clone_33 = None
    bmm_13: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_513, view_514)
    view_515: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_13, [12, 4, 256, 1, 64]);  bmm_13 = None
    permute_434: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_515, [0, 1, 2, 4, 3]);  view_515 = None
    view_516: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_434, [12, 4, 256, 64]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_517: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_516, [1, 12, 1024, 64]);  view_516 = None
    permute_435: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_436: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_435, [1, 0, 2, 3]);  permute_435 = None
    clone_34: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    view_518: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_34, [1024, 1, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_437: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_518, [1, 0, 2]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_519: "f32[1024, 768]" = torch.ops.aten.view.default(permute_437, [1024, 768]);  permute_437 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_104, view_519, permute_438);  primals_104 = None
    view_520: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_19 = torch.ops.aten.native_dropout.default(view_520, 0.1, True);  view_520 = None
    getitem_62: "f32[1, 1024, 768]" = native_dropout_19[0]
    getitem_63: "b8[1, 1024, 768]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_70: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_62, add_65);  getitem_62 = add_65 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_65: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_54: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_65);  add_70 = getitem_65 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_105)
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_106);  mul_50 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_521: "f32[1024, 768]" = torch.ops.aten.view.default(add_72, [1024, 768])
    permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_40: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_108, view_521, permute_439);  primals_108 = None
    view_522: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_51: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, 0.5)
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, 0.7071067811865476);  view_522 = None
    erf_6: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_73: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_53: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_73);  mul_51 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_523: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_53, [1024, 3072]);  mul_53 = None
    permute_440: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_110, view_523, permute_440);  primals_110 = None
    view_524: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_524, 0.1, True);  view_524 = None
    getitem_66: "f32[1, 1024, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 1024, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_72);  getitem_66 = add_72 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_75: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_55: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_69);  add_74 = getitem_69 = None
    mul_54: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = None
    mul_55: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_111)
    add_76: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_112);  mul_55 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_441: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_76, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_525: "f32[1024, 768]" = torch.ops.aten.view.default(permute_441, [1024, 768]);  permute_441 = None
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_42: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_114, view_525, permute_442);  primals_114 = None
    view_526: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_42, [1024, 1, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_116, view_525, permute_443);  primals_116 = None
    view_528: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_43, [1024, 1, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_44: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_118, view_525, permute_444);  primals_118 = None
    view_530: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_44, [1024, 1, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_70: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_526, 8.0);  view_526 = None
    view_531: "f32[1024, 768]" = torch.ops.aten.view.default(div_70, [1024, 768]);  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_534: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_528, [1024, 1, 12, 64]);  view_528 = None
    permute_446: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_534, [1, 0, 2, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_448: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_446, [0, 2, 1, 3]);  permute_446 = None
    view_536: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_448, [12, 1024, 64]);  permute_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_538: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_536, [12, 2, 512, 64]);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_43: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_538, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_134: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
    permute_450: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_134, [0, 1, 4, 2, 3]);  unsqueeze_134 = None
    view_539: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_531, [1024, 1, 768]);  view_531 = None
    view_540: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_539, [1024, 1, 12, 64]);  view_539 = None
    permute_452: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_540, [1, 0, 2, 3]);  view_540 = None
    permute_453: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
    view_541: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_453, [12, 1024, 64]);  permute_453 = None
    view_542: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_541, [12, 2, 512, 64]);  view_541 = None
    as_strided_44: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_542, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_542 = None
    unsqueeze_135: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
    permute_454: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_135, [0, 1, 2, 4, 3]);  unsqueeze_135 = None
    permute_455: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_454, [0, 1, 2, 4, 3]);  permute_454 = None
    clone_35: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    view_543: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_35, [36, 512, 64]);  clone_35 = None
    permute_456: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_450, [0, 1, 4, 3, 2]);  permute_450 = None
    clone_36: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
    view_544: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_36, [36, 64, 512]);  clone_36 = None
    bmm_14: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_543, view_544)
    view_545: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_14, [12, 3, 512, 1, 512]);  bmm_14 = None
    permute_457: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_545, [0, 1, 2, 4, 3]);  view_545 = None
    view_546: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_457, [12, 3, 512, 512]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_28: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_546, [0, 0, 0, 1], 0.0);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_547: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_28, [12, 3, 512, 513]);  constant_pad_nd_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1618: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_547, 0, 0, 9223372036854775807);  view_547 = None
    slice_1619: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1618, 1, 0, 9223372036854775807)
    slice_1620: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1619, 2, 0, 256)
    slice_1621: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257);  slice_1620 = None
    copy_84: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_1621);  slice_1621 = None
    slice_scatter_308: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_84, 3, 256, 9223372036854775807);  copy_84 = None
    slice_scatter_309: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_308, 2, 0, 9223372036854775807);  slice_scatter_308 = None
    slice_scatter_310: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_309, 1, 0, -1);  slice_scatter_309 = None
    slice_scatter_311: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_310, 0, 0, 9223372036854775807);  slice_scatter_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_140: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1618, 1, -1)
    slice_1634: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_140, 1, 256, 9223372036854775807);  select_140 = None
    slice_1635: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1634, 2, 0, 257);  slice_1634 = None
    slice_1639: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_311, 0, 0, 9223372036854775807)
    select_142: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1639, 1, -1)
    slice_1640: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_142, 1, 0, 9223372036854775807)
    slice_1641: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1640, 2, 256, 9223372036854775807)
    copy_85: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1641, slice_1635);  slice_1641 = slice_1635 = None
    slice_scatter_312: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1640, copy_85, 2, 256, 9223372036854775807);  slice_1640 = copy_85 = None
    slice_scatter_313: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_142, slice_scatter_312, 1, 0, 9223372036854775807);  select_142 = slice_scatter_312 = None
    select_scatter_28: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1639, slice_scatter_313, 1, -1);  slice_1639 = slice_scatter_313 = None
    slice_scatter_314: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_311, select_scatter_28, 0, 0, 9223372036854775807);  slice_scatter_311 = select_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1649: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1619, 2, -257, -1);  slice_1619 = None
    slice_1650: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1649, 3, 257, 9223372036854775807);  slice_1649 = None
    slice_1655: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_314, 0, 0, 9223372036854775807)
    slice_1656: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1655, 1, 1, 9223372036854775807)
    slice_1657: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1656, 2, 0, 9223372036854775807)
    slice_1658: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1657, 3, 0, 256)
    copy_86: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1658, slice_1650);  slice_1658 = slice_1650 = None
    slice_scatter_315: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1657, copy_86, 3, 0, 256);  slice_1657 = copy_86 = None
    slice_scatter_316: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1656, slice_scatter_315, 2, 0, 9223372036854775807);  slice_1656 = slice_scatter_315 = None
    slice_scatter_317: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1655, slice_scatter_316, 1, 1, 9223372036854775807);  slice_1655 = slice_scatter_316 = None
    slice_scatter_318: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_314, slice_scatter_317, 0, 0, 9223372036854775807);  slice_scatter_314 = slice_scatter_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_145: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1618, 1, 0);  slice_1618 = None
    slice_1667: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_145, 1, 0, 255);  select_145 = None
    slice_1668: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1667, 2, -255, 9223372036854775807);  slice_1667 = None
    slice_1672: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_318, 0, 0, 9223372036854775807)
    select_147: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1672, 1, 0)
    slice_1673: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_147, 1, 1, 256)
    slice_1674: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1673, 2, 1, 256)
    copy_87: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1674, slice_1668);  slice_1674 = slice_1668 = None
    slice_scatter_319: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1673, copy_87, 2, 1, 256);  slice_1673 = copy_87 = None
    slice_scatter_320: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_147, slice_scatter_319, 1, 1, 256);  select_147 = slice_scatter_319 = None
    select_scatter_29: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1672, slice_scatter_320, 1, 0);  slice_1672 = slice_scatter_320 = None
    slice_scatter_321: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_318, select_scatter_29, 0, 0, 9223372036854775807);  slice_scatter_318 = select_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_550: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_321, [1, 12, 1024, 513]);  slice_scatter_321 = None
    permute_460: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
    slice_1686: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_460, 0, 0, 9223372036854775807)
    slice_1687: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1686, 1, 0, 256)
    slice_1688: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1687, 2, 0, 9223372036854775807)
    slice_1689: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1688, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_57: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_1689)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_88: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1689, where_57);  slice_1689 = where_57 = None
    slice_scatter_322: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1688, copy_88, 3, 0, 257);  slice_1688 = copy_88 = None
    slice_scatter_323: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1687, slice_scatter_322, 2, 0, 9223372036854775807);  slice_1687 = slice_scatter_322 = None
    slice_scatter_324: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1686, slice_scatter_323, 1, 0, 256);  slice_1686 = slice_scatter_323 = None
    slice_scatter_325: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_460, slice_scatter_324, 0, 0, 9223372036854775807);  permute_460 = slice_scatter_324 = None
    permute_463: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_325, [0, 2, 1, 3]);  slice_scatter_325 = None
    view_553: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_463, [12, 4, 256, 513]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_555: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_553, [1, 12, 1024, 513]);  view_553 = None
    permute_465: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    slice_1709: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_465, 0, 0, 9223372036854775807)
    slice_1710: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1709, 1, -256, 9223372036854775807)
    slice_1711: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1710, 2, 0, 9223372036854775807)
    slice_1712: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1711, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_58: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_1712)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_89: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1712, where_58);  slice_1712 = where_58 = None
    slice_scatter_326: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1711, copy_89, 3, -257, 9223372036854775807);  slice_1711 = copy_89 = None
    slice_scatter_327: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1710, slice_scatter_326, 2, 0, 9223372036854775807);  slice_1710 = slice_scatter_326 = None
    slice_scatter_328: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1709, slice_scatter_327, 1, -256, 9223372036854775807);  slice_1709 = slice_scatter_327 = None
    slice_scatter_329: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_465, slice_scatter_328, 0, 0, 9223372036854775807);  permute_465 = slice_scatter_328 = None
    permute_468: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_329, [0, 2, 1, 3]);  slice_scatter_329 = None
    view_558: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_468, [12, 4, 256, 513]);  permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_578: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_558, [1, 12, 1024, 513]);  view_558 = None
    permute_486: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
    add_79: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_486, permute_46);  permute_486 = None
    permute_488: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_79, [0, 2, 1, 3]);  add_79 = None
    view_581: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_488, [12, 4, 256, 513]);  permute_488 = None
    view_582: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_581, [1, 12, 1024, 513]);  view_581 = None
    permute_489: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_37: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    amax_7: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_37, [-1], True)
    sub_60: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_37, amax_7);  clone_37 = amax_7 = None
    exp_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_8: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_77: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_63: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_77);  div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_21 = torch.ops.aten.native_dropout.default(where_63, 0.1, True);  where_63 = None
    getitem_70: "f32[1, 1024, 12, 513]" = native_dropout_21[0]
    getitem_71: "b8[1, 1024, 12, 513]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_583: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_530, [1024, 1, 12, 64]);  view_530 = None
    permute_490: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_583, [1, 0, 2, 3]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_491: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
    view_584: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_491, [12, 4, 256, 513]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_492: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
    view_585: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_492, [12, 1024, 64]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_30: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_585, [0, 0, 256, 256], -1.0);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_47: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_31: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_584, [0, 257], 0.0);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_586: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_31, [12, 4, -1]);  constant_pad_nd_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1842: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_586, 0, 0, 9223372036854775807);  view_586 = None
    slice_1843: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1842, 1, 0, 9223372036854775807);  slice_1842 = None
    slice_1844: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1843, 2, 0, -256);  slice_1843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_587: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1844, [12, 4, 256, 769]);  slice_1844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1845: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_587, 0, 0, 9223372036854775807);  view_587 = None
    slice_1846: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1845, 1, 0, 9223372036854775807);  slice_1845 = None
    slice_1847: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1846, 2, 0, 9223372036854775807);  slice_1846 = None
    slice_1848: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1847, 3, 0, -1);  slice_1847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_150: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1848, 4);  slice_1848 = None
    permute_493: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 2, 4, 3]);  unsqueeze_150 = None
    unsqueeze_151: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_47, 4);  as_strided_47 = None
    permute_494: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_151, [0, 1, 4, 3, 2]);  unsqueeze_151 = None
    permute_495: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_493, [0, 1, 2, 4, 3]);  permute_493 = None
    view_588: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_495, [48, 256, 768]);  permute_495 = None
    permute_496: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_494, [0, 1, 4, 3, 2]);  permute_494 = None
    clone_38: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
    view_589: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_38, [48, 768, 64]);  clone_38 = None
    bmm_15: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_588, view_589)
    view_590: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_15, [12, 4, 256, 1, 64]);  bmm_15 = None
    permute_497: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_590, [0, 1, 2, 4, 3]);  view_590 = None
    view_591: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_497, [12, 4, 256, 64]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_592: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_591, [1, 12, 1024, 64]);  view_591 = None
    permute_498: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_499: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_498, [1, 0, 2, 3]);  permute_498 = None
    clone_39: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_593: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_39, [1024, 1, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_500: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_593, [1, 0, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_594: "f32[1024, 768]" = torch.ops.aten.view.default(permute_500, [1024, 768]);  permute_500 = None
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_120, view_594, permute_501);  primals_120 = None
    view_595: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_22 = torch.ops.aten.native_dropout.default(view_595, 0.1, True);  view_595 = None
    getitem_72: "f32[1, 1024, 768]" = native_dropout_22[0]
    getitem_73: "b8[1, 1024, 768]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_72, add_76);  getitem_72 = add_76 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_75: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_82: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_62: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_75);  add_81 = getitem_75 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = None
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_121)
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_122);  mul_58 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_596: "f32[1024, 768]" = torch.ops.aten.view.default(add_83, [1024, 768])
    permute_502: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_124, view_596, permute_502);  primals_124 = None
    view_597: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_59: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, 0.5)
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, 0.7071067811865476);  view_597 = None
    erf_7: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_84: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_84);  mul_59 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_598: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_61, [1024, 3072]);  mul_61 = None
    permute_503: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_126, view_598, permute_503);  primals_126 = None
    view_599: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_599, 0.1, True);  view_599 = None
    getitem_76: "f32[1, 1024, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 1024, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_83);  getitem_76 = add_83 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_86: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_63: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_79);  add_85 = getitem_79 = None
    mul_62: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = None
    mul_63: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, primals_127)
    add_87: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, primals_128);  mul_63 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_504: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_87, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_600: "f32[1024, 768]" = torch.ops.aten.view.default(permute_504, [1024, 768]);  permute_504 = None
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_48: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_130, view_600, permute_505);  primals_130 = None
    view_601: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_48, [1024, 1, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_49: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_132, view_600, permute_506);  primals_132 = None
    view_603: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_49, [1024, 1, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_507: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_50: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_134, view_600, permute_507);  primals_134 = None
    view_605: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_50, [1024, 1, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_80: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_601, 8.0);  view_601 = None
    view_606: "f32[1024, 768]" = torch.ops.aten.view.default(div_80, [1024, 768]);  div_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_609: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_603, [1024, 1, 12, 64]);  view_603 = None
    permute_509: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_609, [1, 0, 2, 3]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_511: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_509, [0, 2, 1, 3]);  permute_509 = None
    view_611: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_511, [12, 1024, 64]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_613: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_611, [12, 2, 512, 64]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_49: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_613, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_153: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_49, 4);  as_strided_49 = None
    permute_513: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_153, [0, 1, 4, 2, 3]);  unsqueeze_153 = None
    view_614: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_606, [1024, 1, 768]);  view_606 = None
    view_615: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_614, [1024, 1, 12, 64]);  view_614 = None
    permute_515: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_615, [1, 0, 2, 3]);  view_615 = None
    permute_516: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_515, [0, 2, 1, 3]);  permute_515 = None
    view_616: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_516, [12, 1024, 64]);  permute_516 = None
    view_617: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_616, [12, 2, 512, 64]);  view_616 = None
    as_strided_50: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_617, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_617 = None
    unsqueeze_154: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
    permute_517: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 2, 4, 3]);  unsqueeze_154 = None
    permute_518: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_517, [0, 1, 2, 4, 3]);  permute_517 = None
    clone_40: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_618: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_40, [36, 512, 64]);  clone_40 = None
    permute_519: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_513, [0, 1, 4, 3, 2]);  permute_513 = None
    clone_41: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_619: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_41, [36, 64, 512]);  clone_41 = None
    bmm_16: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_618, view_619)
    view_620: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_16, [12, 3, 512, 1, 512]);  bmm_16 = None
    permute_520: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_620, [0, 1, 2, 4, 3]);  view_620 = None
    view_621: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_520, [12, 3, 512, 512]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_32: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_621, [0, 0, 0, 1], 0.0);  view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_622: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_32, [12, 3, 512, 513]);  constant_pad_nd_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1849: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_622, 0, 0, 9223372036854775807);  view_622 = None
    slice_1850: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1849, 1, 0, 9223372036854775807)
    slice_1851: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1850, 2, 0, 256)
    slice_1852: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1851, 3, 0, 257);  slice_1851 = None
    copy_96: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_1852);  slice_1852 = None
    slice_scatter_352: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_96, 3, 256, 9223372036854775807);  copy_96 = None
    slice_scatter_353: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_352, 2, 0, 9223372036854775807);  slice_scatter_352 = None
    slice_scatter_354: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_353, 1, 0, -1);  slice_scatter_353 = None
    slice_scatter_355: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_354, 0, 0, 9223372036854775807);  slice_scatter_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_160: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1849, 1, -1)
    slice_1865: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_160, 1, 256, 9223372036854775807);  select_160 = None
    slice_1866: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1865, 2, 0, 257);  slice_1865 = None
    slice_1870: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_355, 0, 0, 9223372036854775807)
    select_162: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1870, 1, -1)
    slice_1871: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_162, 1, 0, 9223372036854775807)
    slice_1872: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1871, 2, 256, 9223372036854775807)
    copy_97: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1872, slice_1866);  slice_1872 = slice_1866 = None
    slice_scatter_356: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1871, copy_97, 2, 256, 9223372036854775807);  slice_1871 = copy_97 = None
    slice_scatter_357: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_162, slice_scatter_356, 1, 0, 9223372036854775807);  select_162 = slice_scatter_356 = None
    select_scatter_32: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1870, slice_scatter_357, 1, -1);  slice_1870 = slice_scatter_357 = None
    slice_scatter_358: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_355, select_scatter_32, 0, 0, 9223372036854775807);  slice_scatter_355 = select_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1880: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1850, 2, -257, -1);  slice_1850 = None
    slice_1881: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1880, 3, 257, 9223372036854775807);  slice_1880 = None
    slice_1886: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_358, 0, 0, 9223372036854775807)
    slice_1887: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1886, 1, 1, 9223372036854775807)
    slice_1888: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1887, 2, 0, 9223372036854775807)
    slice_1889: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1888, 3, 0, 256)
    copy_98: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1889, slice_1881);  slice_1889 = slice_1881 = None
    slice_scatter_359: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1888, copy_98, 3, 0, 256);  slice_1888 = copy_98 = None
    slice_scatter_360: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1887, slice_scatter_359, 2, 0, 9223372036854775807);  slice_1887 = slice_scatter_359 = None
    slice_scatter_361: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1886, slice_scatter_360, 1, 1, 9223372036854775807);  slice_1886 = slice_scatter_360 = None
    slice_scatter_362: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_358, slice_scatter_361, 0, 0, 9223372036854775807);  slice_scatter_358 = slice_scatter_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_165: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1849, 1, 0);  slice_1849 = None
    slice_1898: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_165, 1, 0, 255);  select_165 = None
    slice_1899: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1898, 2, -255, 9223372036854775807);  slice_1898 = None
    slice_1903: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_362, 0, 0, 9223372036854775807)
    select_167: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1903, 1, 0)
    slice_1904: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256)
    slice_1905: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1904, 2, 1, 256)
    copy_99: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1905, slice_1899);  slice_1905 = slice_1899 = None
    slice_scatter_363: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1904, copy_99, 2, 1, 256);  slice_1904 = copy_99 = None
    slice_scatter_364: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_167, slice_scatter_363, 1, 1, 256);  select_167 = slice_scatter_363 = None
    select_scatter_33: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1903, slice_scatter_364, 1, 0);  slice_1903 = slice_scatter_364 = None
    slice_scatter_365: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_362, select_scatter_33, 0, 0, 9223372036854775807);  slice_scatter_362 = select_scatter_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_625: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_365, [1, 12, 1024, 513]);  slice_scatter_365 = None
    permute_523: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_625, [0, 2, 1, 3]);  view_625 = None
    slice_1917: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_523, 0, 0, 9223372036854775807)
    slice_1918: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1917, 1, 0, 256)
    slice_1919: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1918, 2, 0, 9223372036854775807)
    slice_1920: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1919, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_65: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_100: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1920, where_65);  slice_1920 = where_65 = None
    slice_scatter_366: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1919, copy_100, 3, 0, 257);  slice_1919 = copy_100 = None
    slice_scatter_367: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1918, slice_scatter_366, 2, 0, 9223372036854775807);  slice_1918 = slice_scatter_366 = None
    slice_scatter_368: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1917, slice_scatter_367, 1, 0, 256);  slice_1917 = slice_scatter_367 = None
    slice_scatter_369: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_523, slice_scatter_368, 0, 0, 9223372036854775807);  permute_523 = slice_scatter_368 = None
    permute_526: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_369, [0, 2, 1, 3]);  slice_scatter_369 = None
    view_628: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_526, [12, 4, 256, 513]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_630: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_628, [1, 12, 1024, 513]);  view_628 = None
    permute_528: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    slice_1940: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_528, 0, 0, 9223372036854775807)
    slice_1941: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1940, 1, -256, 9223372036854775807)
    slice_1942: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1941, 2, 0, 9223372036854775807)
    slice_1943: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1942, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_66: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_1943)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_101: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1943, where_66);  slice_1943 = where_66 = None
    slice_scatter_370: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1942, copy_101, 3, -257, 9223372036854775807);  slice_1942 = copy_101 = None
    slice_scatter_371: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1941, slice_scatter_370, 2, 0, 9223372036854775807);  slice_1941 = slice_scatter_370 = None
    slice_scatter_372: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1940, slice_scatter_371, 1, -256, 9223372036854775807);  slice_1940 = slice_scatter_371 = None
    slice_scatter_373: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_528, slice_scatter_372, 0, 0, 9223372036854775807);  permute_528 = slice_scatter_372 = None
    permute_531: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_373, [0, 2, 1, 3]);  slice_scatter_373 = None
    view_633: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_531, [12, 4, 256, 513]);  permute_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_653: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_633, [1, 12, 1024, 513]);  view_633 = None
    permute_549: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    add_90: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_549, permute_46);  permute_549 = None
    permute_551: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_90, [0, 2, 1, 3]);  add_90 = None
    view_656: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_551, [12, 4, 256, 513]);  permute_551 = None
    view_657: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_656, [1, 12, 1024, 513]);  view_656 = None
    permute_552: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_42: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    amax_8: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_42, [-1], True)
    sub_68: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_42, amax_8);  clone_42 = amax_8 = None
    exp_8: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_9: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_71: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_87);  div_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_24 = torch.ops.aten.native_dropout.default(where_71, 0.1, True);  where_71 = None
    getitem_80: "f32[1, 1024, 12, 513]" = native_dropout_24[0]
    getitem_81: "b8[1, 1024, 12, 513]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_658: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_605, [1024, 1, 12, 64]);  view_605 = None
    permute_553: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_658, [1, 0, 2, 3]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_554: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3]);  getitem_80 = None
    view_659: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_554, [12, 4, 256, 513]);  permute_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_555: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_553, [0, 2, 1, 3]);  permute_553 = None
    view_660: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_555, [12, 1024, 64]);  permute_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_34: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_660, [0, 0, 256, 256], -1.0);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_53: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_35: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_659, [0, 257], 0.0);  view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_661: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_35, [12, 4, -1]);  constant_pad_nd_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2073: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_661, 0, 0, 9223372036854775807);  view_661 = None
    slice_2074: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2073, 1, 0, 9223372036854775807);  slice_2073 = None
    slice_2075: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2074, 2, 0, -256);  slice_2074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_662: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2075, [12, 4, 256, 769]);  slice_2075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2076: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_662, 0, 0, 9223372036854775807);  view_662 = None
    slice_2077: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2076, 1, 0, 9223372036854775807);  slice_2076 = None
    slice_2078: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2077, 2, 0, 9223372036854775807);  slice_2077 = None
    slice_2079: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2078, 3, 0, -1);  slice_2078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_169: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2079, 4);  slice_2079 = None
    permute_556: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_169, [0, 1, 2, 4, 3]);  unsqueeze_169 = None
    unsqueeze_170: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
    permute_557: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 3, 2]);  unsqueeze_170 = None
    permute_558: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_556, [0, 1, 2, 4, 3]);  permute_556 = None
    view_663: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_558, [48, 256, 768]);  permute_558 = None
    permute_559: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_557, [0, 1, 4, 3, 2]);  permute_557 = None
    clone_43: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_664: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_43, [48, 768, 64]);  clone_43 = None
    bmm_17: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_663, view_664)
    view_665: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_17, [12, 4, 256, 1, 64]);  bmm_17 = None
    permute_560: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_665, [0, 1, 2, 4, 3]);  view_665 = None
    view_666: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_560, [12, 4, 256, 64]);  permute_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_667: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_666, [1, 12, 1024, 64]);  view_666 = None
    permute_561: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_667, [0, 2, 1, 3]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_562: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_561, [1, 0, 2, 3]);  permute_561 = None
    clone_44: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
    view_668: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_44, [1024, 1, 768]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_563: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_668, [1, 0, 2]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_669: "f32[1024, 768]" = torch.ops.aten.view.default(permute_563, [1024, 768]);  permute_563 = None
    permute_564: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_51: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_136, view_669, permute_564);  primals_136 = None
    view_670: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_51, [1, 1024, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_25 = torch.ops.aten.native_dropout.default(view_670, 0.1, True);  view_670 = None
    getitem_82: "f32[1, 1024, 768]" = native_dropout_25[0]
    getitem_83: "b8[1, 1024, 768]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_92: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_82, add_87);  getitem_82 = add_87 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_85: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_93: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_70: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_85);  add_92 = getitem_85 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = None
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_137)
    add_94: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_138);  mul_66 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_671: "f32[1024, 768]" = torch.ops.aten.view.default(add_94, [1024, 768])
    permute_565: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_52: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_140, view_671, permute_565);  primals_140 = None
    view_672: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_67: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, 0.5)
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, 0.7071067811865476);  view_672 = None
    erf_8: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_69: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_95);  mul_67 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_69, [1024, 3072]);  mul_69 = None
    permute_566: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_53: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_142, view_673, permute_566);  primals_142 = None
    view_674: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_53, [1, 1024, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_674, 0.1, True);  view_674 = None
    getitem_86: "f32[1, 1024, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 1024, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_94);  getitem_86 = add_94 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_71: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_89);  add_96 = getitem_89 = None
    mul_70: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = None
    mul_71: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_143)
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_144);  mul_71 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_567: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_98, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_675: "f32[1024, 768]" = torch.ops.aten.view.default(permute_567, [1024, 768]);  permute_567 = None
    permute_568: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_54: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_146, view_675, permute_568);  primals_146 = None
    view_676: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_54, [1024, 1, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_569: "f32[768, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_55: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_148, view_675, permute_569);  primals_148 = None
    view_678: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_55, [1024, 1, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_570: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_56: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_150, view_675, permute_570);  primals_150 = None
    view_680: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_56, [1024, 1, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_90: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_676, 8.0);  view_676 = None
    view_681: "f32[1024, 768]" = torch.ops.aten.view.default(div_90, [1024, 768]);  div_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_684: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_678, [1024, 1, 12, 64]);  view_678 = None
    permute_572: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_684, [1, 0, 2, 3]);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_574: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_572, [0, 2, 1, 3]);  permute_572 = None
    view_686: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_574, [12, 1024, 64]);  permute_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_688: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_686, [12, 2, 512, 64]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_55: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_688, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_172: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
    permute_576: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_172, [0, 1, 4, 2, 3]);  unsqueeze_172 = None
    view_689: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_681, [1024, 1, 768]);  view_681 = None
    view_690: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_689, [1024, 1, 12, 64]);  view_689 = None
    permute_578: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_690, [1, 0, 2, 3]);  view_690 = None
    permute_579: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_578, [0, 2, 1, 3]);  permute_578 = None
    view_691: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_579, [12, 1024, 64]);  permute_579 = None
    view_692: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_691, [12, 2, 512, 64]);  view_691 = None
    as_strided_56: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_692, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_692 = None
    unsqueeze_173: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_56, 4);  as_strided_56 = None
    permute_580: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_173, [0, 1, 2, 4, 3]);  unsqueeze_173 = None
    permute_581: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_580, [0, 1, 2, 4, 3]);  permute_580 = None
    clone_45: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
    view_693: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_45, [36, 512, 64]);  clone_45 = None
    permute_582: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_576, [0, 1, 4, 3, 2]);  permute_576 = None
    clone_46: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
    view_694: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_46, [36, 64, 512]);  clone_46 = None
    bmm_18: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_693, view_694)
    view_695: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, [12, 3, 512, 1, 512]);  bmm_18 = None
    permute_583: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_695, [0, 1, 2, 4, 3]);  view_695 = None
    view_696: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_583, [12, 3, 512, 512]);  permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_36: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_696, [0, 0, 0, 1], 0.0);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_697: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_36, [12, 3, 512, 513]);  constant_pad_nd_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2080: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_697, 0, 0, 9223372036854775807);  view_697 = None
    slice_2081: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2080, 1, 0, 9223372036854775807)
    slice_2082: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2081, 2, 0, 256)
    slice_2083: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2082, 3, 0, 257);  slice_2082 = None
    copy_108: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_2083);  slice_2083 = None
    slice_scatter_396: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_108, 3, 256, 9223372036854775807);  copy_108 = None
    slice_scatter_397: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_396, 2, 0, 9223372036854775807);  slice_scatter_396 = None
    slice_scatter_398: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_397, 1, 0, -1);  slice_scatter_397 = None
    slice_scatter_399: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_398, 0, 0, 9223372036854775807);  slice_scatter_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_180: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2080, 1, -1)
    slice_2096: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_180, 1, 256, 9223372036854775807);  select_180 = None
    slice_2097: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2096, 2, 0, 257);  slice_2096 = None
    slice_2101: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_399, 0, 0, 9223372036854775807)
    select_182: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2101, 1, -1)
    slice_2102: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_182, 1, 0, 9223372036854775807)
    slice_2103: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2102, 2, 256, 9223372036854775807)
    copy_109: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2103, slice_2097);  slice_2103 = slice_2097 = None
    slice_scatter_400: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2102, copy_109, 2, 256, 9223372036854775807);  slice_2102 = copy_109 = None
    slice_scatter_401: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_182, slice_scatter_400, 1, 0, 9223372036854775807);  select_182 = slice_scatter_400 = None
    select_scatter_36: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2101, slice_scatter_401, 1, -1);  slice_2101 = slice_scatter_401 = None
    slice_scatter_402: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_399, select_scatter_36, 0, 0, 9223372036854775807);  slice_scatter_399 = select_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2111: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2081, 2, -257, -1);  slice_2081 = None
    slice_2112: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2111, 3, 257, 9223372036854775807);  slice_2111 = None
    slice_2117: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_402, 0, 0, 9223372036854775807)
    slice_2118: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2117, 1, 1, 9223372036854775807)
    slice_2119: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2118, 2, 0, 9223372036854775807)
    slice_2120: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2119, 3, 0, 256)
    copy_110: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2120, slice_2112);  slice_2120 = slice_2112 = None
    slice_scatter_403: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2119, copy_110, 3, 0, 256);  slice_2119 = copy_110 = None
    slice_scatter_404: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2118, slice_scatter_403, 2, 0, 9223372036854775807);  slice_2118 = slice_scatter_403 = None
    slice_scatter_405: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2117, slice_scatter_404, 1, 1, 9223372036854775807);  slice_2117 = slice_scatter_404 = None
    slice_scatter_406: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_402, slice_scatter_405, 0, 0, 9223372036854775807);  slice_scatter_402 = slice_scatter_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_185: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2080, 1, 0);  slice_2080 = None
    slice_2129: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_185, 1, 0, 255);  select_185 = None
    slice_2130: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2129, 2, -255, 9223372036854775807);  slice_2129 = None
    slice_2134: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_406, 0, 0, 9223372036854775807)
    select_187: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2134, 1, 0)
    slice_2135: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_187, 1, 1, 256)
    slice_2136: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2135, 2, 1, 256)
    copy_111: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2136, slice_2130);  slice_2136 = slice_2130 = None
    slice_scatter_407: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2135, copy_111, 2, 1, 256);  slice_2135 = copy_111 = None
    slice_scatter_408: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_187, slice_scatter_407, 1, 1, 256);  select_187 = slice_scatter_407 = None
    select_scatter_37: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2134, slice_scatter_408, 1, 0);  slice_2134 = slice_scatter_408 = None
    slice_scatter_409: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_406, select_scatter_37, 0, 0, 9223372036854775807);  slice_scatter_406 = select_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_700: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_409, [1, 12, 1024, 513]);  slice_scatter_409 = None
    permute_586: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
    slice_2148: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_586, 0, 0, 9223372036854775807)
    slice_2149: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2148, 1, 0, 256)
    slice_2150: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2149, 2, 0, 9223372036854775807)
    slice_2151: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2150, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_73: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_2151)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_112: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2151, where_73);  slice_2151 = where_73 = None
    slice_scatter_410: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2150, copy_112, 3, 0, 257);  slice_2150 = copy_112 = None
    slice_scatter_411: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2149, slice_scatter_410, 2, 0, 9223372036854775807);  slice_2149 = slice_scatter_410 = None
    slice_scatter_412: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2148, slice_scatter_411, 1, 0, 256);  slice_2148 = slice_scatter_411 = None
    slice_scatter_413: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_586, slice_scatter_412, 0, 0, 9223372036854775807);  permute_586 = slice_scatter_412 = None
    permute_589: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_413, [0, 2, 1, 3]);  slice_scatter_413 = None
    view_703: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_589, [12, 4, 256, 513]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_705: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_703, [1, 12, 1024, 513]);  view_703 = None
    permute_591: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_705, [0, 2, 1, 3]);  view_705 = None
    slice_2171: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_591, 0, 0, 9223372036854775807)
    slice_2172: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2171, 1, -256, 9223372036854775807)
    slice_2173: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2172, 2, 0, 9223372036854775807)
    slice_2174: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2173, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_74: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_2174)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_113: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2174, where_74);  slice_2174 = where_74 = None
    slice_scatter_414: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2173, copy_113, 3, -257, 9223372036854775807);  slice_2173 = copy_113 = None
    slice_scatter_415: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2172, slice_scatter_414, 2, 0, 9223372036854775807);  slice_2172 = slice_scatter_414 = None
    slice_scatter_416: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2171, slice_scatter_415, 1, -256, 9223372036854775807);  slice_2171 = slice_scatter_415 = None
    slice_scatter_417: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_591, slice_scatter_416, 0, 0, 9223372036854775807);  permute_591 = slice_scatter_416 = None
    permute_594: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_417, [0, 2, 1, 3]);  slice_scatter_417 = None
    view_708: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_594, [12, 4, 256, 513]);  permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_728: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_708, [1, 12, 1024, 513]);  view_708 = None
    permute_612: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_728, [0, 2, 1, 3]);  view_728 = None
    add_101: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_612, permute_46);  permute_612 = None
    permute_614: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_101, [0, 2, 1, 3]);  add_101 = None
    view_731: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_614, [12, 4, 256, 513]);  permute_614 = None
    view_732: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_731, [1, 12, 1024, 513]);  view_731 = None
    permute_615: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_47: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    amax_9: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_47, [-1], True)
    sub_76: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_47, amax_9);  clone_47 = amax_9 = None
    exp_9: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_10: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_97: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_79: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_97);  div_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_27 = torch.ops.aten.native_dropout.default(where_79, 0.1, True);  where_79 = None
    getitem_90: "f32[1, 1024, 12, 513]" = native_dropout_27[0]
    getitem_91: "b8[1, 1024, 12, 513]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_733: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_680, [1024, 1, 12, 64]);  view_680 = None
    permute_616: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_733, [1, 0, 2, 3]);  view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_617: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
    view_734: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_617, [12, 4, 256, 513]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_618: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_616, [0, 2, 1, 3]);  permute_616 = None
    view_735: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_618, [12, 1024, 64]);  permute_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_38: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_735, [0, 0, 256, 256], -1.0);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_59: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_39: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_734, [0, 257], 0.0);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_736: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_39, [12, 4, -1]);  constant_pad_nd_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2304: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_736, 0, 0, 9223372036854775807);  view_736 = None
    slice_2305: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2304, 1, 0, 9223372036854775807);  slice_2304 = None
    slice_2306: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2305, 2, 0, -256);  slice_2305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_737: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2306, [12, 4, 256, 769]);  slice_2306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2307: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_737, 0, 0, 9223372036854775807);  view_737 = None
    slice_2308: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2307, 1, 0, 9223372036854775807);  slice_2307 = None
    slice_2309: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2308, 2, 0, 9223372036854775807);  slice_2308 = None
    slice_2310: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2309, 3, 0, -1);  slice_2309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_188: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2310, 4);  slice_2310 = None
    permute_619: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_188, [0, 1, 2, 4, 3]);  unsqueeze_188 = None
    unsqueeze_189: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
    permute_620: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_189, [0, 1, 4, 3, 2]);  unsqueeze_189 = None
    permute_621: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_619, [0, 1, 2, 4, 3]);  permute_619 = None
    view_738: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_621, [48, 256, 768]);  permute_621 = None
    permute_622: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_620, [0, 1, 4, 3, 2]);  permute_620 = None
    clone_48: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
    view_739: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_48, [48, 768, 64]);  clone_48 = None
    bmm_19: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_738, view_739)
    view_740: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_19, [12, 4, 256, 1, 64]);  bmm_19 = None
    permute_623: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
    view_741: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_623, [12, 4, 256, 64]);  permute_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_742: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_741, [1, 12, 1024, 64]);  view_741 = None
    permute_624: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_625: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_624, [1, 0, 2, 3]);  permute_624 = None
    clone_49: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_625, memory_format = torch.contiguous_format);  permute_625 = None
    view_743: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_49, [1024, 1, 768]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_626: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_743, [1, 0, 2]);  view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_744: "f32[1024, 768]" = torch.ops.aten.view.default(permute_626, [1024, 768]);  permute_626 = None
    permute_627: "f32[768, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_57: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_152, view_744, permute_627);  primals_152 = None
    view_745: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_57, [1, 1024, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_28 = torch.ops.aten.native_dropout.default(view_745, 0.1, True);  view_745 = None
    getitem_92: "f32[1, 1024, 768]" = native_dropout_28[0]
    getitem_93: "b8[1, 1024, 768]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_103: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_92, add_98);  getitem_92 = add_98 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_95: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_104: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_78: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_95);  add_103 = getitem_95 = None
    mul_73: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = None
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_153)
    add_105: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_154);  mul_74 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_746: "f32[1024, 768]" = torch.ops.aten.view.default(add_105, [1024, 768])
    permute_628: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_58: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_156, view_746, permute_628);  primals_156 = None
    view_747: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, 0.5)
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, 0.7071067811865476);  view_747 = None
    erf_9: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_106: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_77: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_106);  mul_75 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_748: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_77, [1024, 3072]);  mul_77 = None
    permute_629: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_59: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_158, view_748, permute_629);  primals_158 = None
    view_749: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_59, [1, 1024, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_749, 0.1, True);  view_749 = None
    getitem_96: "f32[1, 1024, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 1024, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_107: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_105);  getitem_96 = add_105 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_108: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_79: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_99);  add_107 = getitem_99 = None
    mul_78: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = None
    mul_79: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, primals_159)
    add_109: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_160);  mul_79 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_630: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_109, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_750: "f32[1024, 768]" = torch.ops.aten.view.default(permute_630, [1024, 768]);  permute_630 = None
    permute_631: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_60: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_162, view_750, permute_631);  primals_162 = None
    view_751: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_60, [1024, 1, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_61: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_164, view_750, permute_632);  primals_164 = None
    view_753: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_61, [1024, 1, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_633: "f32[768, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_62: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_166, view_750, permute_633);  primals_166 = None
    view_755: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_62, [1024, 1, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_100: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_751, 8.0);  view_751 = None
    view_756: "f32[1024, 768]" = torch.ops.aten.view.default(div_100, [1024, 768]);  div_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_759: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_753, [1024, 1, 12, 64]);  view_753 = None
    permute_635: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_759, [1, 0, 2, 3]);  view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_637: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_635, [0, 2, 1, 3]);  permute_635 = None
    view_761: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_637, [12, 1024, 64]);  permute_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_763: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_761, [12, 2, 512, 64]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_61: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_763, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_191: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
    permute_639: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
    view_764: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_756, [1024, 1, 768]);  view_756 = None
    view_765: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_764, [1024, 1, 12, 64]);  view_764 = None
    permute_641: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_765, [1, 0, 2, 3]);  view_765 = None
    permute_642: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_641, [0, 2, 1, 3]);  permute_641 = None
    view_766: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_642, [12, 1024, 64]);  permute_642 = None
    view_767: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_766, [12, 2, 512, 64]);  view_766 = None
    as_strided_62: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_767, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_767 = None
    unsqueeze_192: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
    permute_643: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_192, [0, 1, 2, 4, 3]);  unsqueeze_192 = None
    permute_644: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_643, [0, 1, 2, 4, 3]);  permute_643 = None
    clone_50: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
    view_768: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_50, [36, 512, 64]);  clone_50 = None
    permute_645: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_639, [0, 1, 4, 3, 2]);  permute_639 = None
    clone_51: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_645, memory_format = torch.contiguous_format);  permute_645 = None
    view_769: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_51, [36, 64, 512]);  clone_51 = None
    bmm_20: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_768, view_769)
    view_770: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_20, [12, 3, 512, 1, 512]);  bmm_20 = None
    permute_646: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_770, [0, 1, 2, 4, 3]);  view_770 = None
    view_771: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_646, [12, 3, 512, 512]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_40: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_771, [0, 0, 0, 1], 0.0);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_772: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_40, [12, 3, 512, 513]);  constant_pad_nd_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2311: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_772, 0, 0, 9223372036854775807);  view_772 = None
    slice_2312: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2311, 1, 0, 9223372036854775807)
    slice_2313: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2312, 2, 0, 256)
    slice_2314: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2313, 3, 0, 257);  slice_2313 = None
    copy_120: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_2314);  slice_2314 = None
    slice_scatter_440: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_120, 3, 256, 9223372036854775807);  copy_120 = None
    slice_scatter_441: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_440, 2, 0, 9223372036854775807);  slice_scatter_440 = None
    slice_scatter_442: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_441, 1, 0, -1);  slice_scatter_441 = None
    slice_scatter_443: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_442, 0, 0, 9223372036854775807);  slice_scatter_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_200: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2311, 1, -1)
    slice_2327: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_200, 1, 256, 9223372036854775807);  select_200 = None
    slice_2328: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2327, 2, 0, 257);  slice_2327 = None
    slice_2332: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_443, 0, 0, 9223372036854775807)
    select_202: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2332, 1, -1)
    slice_2333: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_202, 1, 0, 9223372036854775807)
    slice_2334: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2333, 2, 256, 9223372036854775807)
    copy_121: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2334, slice_2328);  slice_2334 = slice_2328 = None
    slice_scatter_444: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2333, copy_121, 2, 256, 9223372036854775807);  slice_2333 = copy_121 = None
    slice_scatter_445: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_202, slice_scatter_444, 1, 0, 9223372036854775807);  select_202 = slice_scatter_444 = None
    select_scatter_40: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2332, slice_scatter_445, 1, -1);  slice_2332 = slice_scatter_445 = None
    slice_scatter_446: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_443, select_scatter_40, 0, 0, 9223372036854775807);  slice_scatter_443 = select_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2342: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2312, 2, -257, -1);  slice_2312 = None
    slice_2343: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2342, 3, 257, 9223372036854775807);  slice_2342 = None
    slice_2348: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_446, 0, 0, 9223372036854775807)
    slice_2349: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2348, 1, 1, 9223372036854775807)
    slice_2350: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2349, 2, 0, 9223372036854775807)
    slice_2351: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2350, 3, 0, 256)
    copy_122: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2351, slice_2343);  slice_2351 = slice_2343 = None
    slice_scatter_447: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2350, copy_122, 3, 0, 256);  slice_2350 = copy_122 = None
    slice_scatter_448: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2349, slice_scatter_447, 2, 0, 9223372036854775807);  slice_2349 = slice_scatter_447 = None
    slice_scatter_449: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2348, slice_scatter_448, 1, 1, 9223372036854775807);  slice_2348 = slice_scatter_448 = None
    slice_scatter_450: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_446, slice_scatter_449, 0, 0, 9223372036854775807);  slice_scatter_446 = slice_scatter_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_205: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2311, 1, 0);  slice_2311 = None
    slice_2360: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_205, 1, 0, 255);  select_205 = None
    slice_2361: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2360, 2, -255, 9223372036854775807);  slice_2360 = None
    slice_2365: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_450, 0, 0, 9223372036854775807)
    select_207: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2365, 1, 0)
    slice_2366: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_207, 1, 1, 256)
    slice_2367: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2366, 2, 1, 256)
    copy_123: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2367, slice_2361);  slice_2367 = slice_2361 = None
    slice_scatter_451: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2366, copy_123, 2, 1, 256);  slice_2366 = copy_123 = None
    slice_scatter_452: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_207, slice_scatter_451, 1, 1, 256);  select_207 = slice_scatter_451 = None
    select_scatter_41: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2365, slice_scatter_452, 1, 0);  slice_2365 = slice_scatter_452 = None
    slice_scatter_453: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_450, select_scatter_41, 0, 0, 9223372036854775807);  slice_scatter_450 = select_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_775: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_453, [1, 12, 1024, 513]);  slice_scatter_453 = None
    permute_649: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_775, [0, 2, 1, 3]);  view_775 = None
    slice_2379: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_649, 0, 0, 9223372036854775807)
    slice_2380: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2379, 1, 0, 256)
    slice_2381: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2380, 2, 0, 9223372036854775807)
    slice_2382: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2381, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_81: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_2382)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_124: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2382, where_81);  slice_2382 = where_81 = None
    slice_scatter_454: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2381, copy_124, 3, 0, 257);  slice_2381 = copy_124 = None
    slice_scatter_455: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2380, slice_scatter_454, 2, 0, 9223372036854775807);  slice_2380 = slice_scatter_454 = None
    slice_scatter_456: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2379, slice_scatter_455, 1, 0, 256);  slice_2379 = slice_scatter_455 = None
    slice_scatter_457: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_649, slice_scatter_456, 0, 0, 9223372036854775807);  permute_649 = slice_scatter_456 = None
    permute_652: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_457, [0, 2, 1, 3]);  slice_scatter_457 = None
    view_778: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_652, [12, 4, 256, 513]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_780: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_778, [1, 12, 1024, 513]);  view_778 = None
    permute_654: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
    slice_2402: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_654, 0, 0, 9223372036854775807)
    slice_2403: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2402, 1, -256, 9223372036854775807)
    slice_2404: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2403, 2, 0, 9223372036854775807)
    slice_2405: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2404, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_82: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_2405)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_125: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2405, where_82);  slice_2405 = where_82 = None
    slice_scatter_458: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2404, copy_125, 3, -257, 9223372036854775807);  slice_2404 = copy_125 = None
    slice_scatter_459: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2403, slice_scatter_458, 2, 0, 9223372036854775807);  slice_2403 = slice_scatter_458 = None
    slice_scatter_460: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2402, slice_scatter_459, 1, -256, 9223372036854775807);  slice_2402 = slice_scatter_459 = None
    slice_scatter_461: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_654, slice_scatter_460, 0, 0, 9223372036854775807);  permute_654 = slice_scatter_460 = None
    permute_657: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_461, [0, 2, 1, 3]);  slice_scatter_461 = None
    view_783: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_657, [12, 4, 256, 513]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_803: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_783, [1, 12, 1024, 513]);  view_783 = None
    permute_675: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_803, [0, 2, 1, 3]);  view_803 = None
    add_112: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_675, permute_46);  permute_675 = None
    permute_677: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_112, [0, 2, 1, 3]);  add_112 = None
    view_806: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_677, [12, 4, 256, 513]);  permute_677 = None
    view_807: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_806, [1, 12, 1024, 513]);  view_806 = None
    permute_678: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_807, [0, 2, 1, 3]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_52: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
    amax_10: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_52, [-1], True)
    sub_84: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_52, amax_10);  clone_52 = amax_10 = None
    exp_10: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_11: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_107: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_107);  div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_30 = torch.ops.aten.native_dropout.default(where_87, 0.1, True);  where_87 = None
    getitem_100: "f32[1, 1024, 12, 513]" = native_dropout_30[0]
    getitem_101: "b8[1, 1024, 12, 513]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_808: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_755, [1024, 1, 12, 64]);  view_755 = None
    permute_679: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_808, [1, 0, 2, 3]);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_680: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
    view_809: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_680, [12, 4, 256, 513]);  permute_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_681: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_679, [0, 2, 1, 3]);  permute_679 = None
    view_810: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_681, [12, 1024, 64]);  permute_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_42: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_810, [0, 0, 256, 256], -1.0);  view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_65: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_43: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_809, [0, 257], 0.0);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_811: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_43, [12, 4, -1]);  constant_pad_nd_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2535: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_811, 0, 0, 9223372036854775807);  view_811 = None
    slice_2536: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2535, 1, 0, 9223372036854775807);  slice_2535 = None
    slice_2537: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2536, 2, 0, -256);  slice_2536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_812: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2537, [12, 4, 256, 769]);  slice_2537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2538: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_812, 0, 0, 9223372036854775807);  view_812 = None
    slice_2539: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2538, 1, 0, 9223372036854775807);  slice_2538 = None
    slice_2540: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2539, 2, 0, 9223372036854775807);  slice_2539 = None
    slice_2541: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2540, 3, 0, -1);  slice_2540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_207: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2541, 4);  slice_2541 = None
    permute_682: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_207, [0, 1, 2, 4, 3]);  unsqueeze_207 = None
    unsqueeze_208: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_65, 4);  as_strided_65 = None
    permute_683: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 4, 3, 2]);  unsqueeze_208 = None
    permute_684: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_682, [0, 1, 2, 4, 3]);  permute_682 = None
    view_813: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_684, [48, 256, 768]);  permute_684 = None
    permute_685: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_683, [0, 1, 4, 3, 2]);  permute_683 = None
    clone_53: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_814: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_53, [48, 768, 64]);  clone_53 = None
    bmm_21: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_813, view_814)
    view_815: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_21, [12, 4, 256, 1, 64]);  bmm_21 = None
    permute_686: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_815, [0, 1, 2, 4, 3]);  view_815 = None
    view_816: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_686, [12, 4, 256, 64]);  permute_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_817: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_816, [1, 12, 1024, 64]);  view_816 = None
    permute_687: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_688: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_687, [1, 0, 2, 3]);  permute_687 = None
    clone_54: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
    view_818: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_54, [1024, 1, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_689: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_818, [1, 0, 2]);  view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_819: "f32[1024, 768]" = torch.ops.aten.view.default(permute_689, [1024, 768]);  permute_689 = None
    permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_63: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_168, view_819, permute_690);  primals_168 = None
    view_820: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_63, [1, 1024, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_31 = torch.ops.aten.native_dropout.default(view_820, 0.1, True);  view_820 = None
    getitem_102: "f32[1, 1024, 768]" = native_dropout_31[0]
    getitem_103: "b8[1, 1024, 768]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_114: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_102, add_109);  getitem_102 = add_109 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_105: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_115: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_86: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_105);  add_114 = getitem_105 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = None
    mul_82: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_169)
    add_116: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_170);  mul_82 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_821: "f32[1024, 768]" = torch.ops.aten.view.default(add_116, [1024, 768])
    permute_691: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_64: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_172, view_821, permute_691);  primals_172 = None
    view_822: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, 0.5)
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, 0.7071067811865476);  view_822 = None
    erf_10: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_117: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_85: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_117);  mul_83 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_823: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_85, [1024, 3072]);  mul_85 = None
    permute_692: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_65: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_174, view_823, permute_692);  primals_174 = None
    view_824: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_65, [1, 1024, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_824, 0.1, True);  view_824 = None
    getitem_106: "f32[1, 1024, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 1024, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_118: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_116);  getitem_106 = add_116 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_119: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_87: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_118, getitem_109);  add_118 = getitem_109 = None
    mul_86: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = None
    mul_87: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, primals_175)
    add_120: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_87, primals_176);  mul_87 = primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_693: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_120, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_825: "f32[1024, 768]" = torch.ops.aten.view.default(permute_693, [1024, 768]);  permute_693 = None
    permute_694: "f32[768, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_66: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_178, view_825, permute_694);  primals_178 = None
    view_826: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_66, [1024, 1, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_695: "f32[768, 768]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_67: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_180, view_825, permute_695);  primals_180 = None
    view_828: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_67, [1024, 1, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_68: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_182, view_825, permute_696);  primals_182 = None
    view_830: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_68, [1024, 1, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_110: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_826, 8.0);  view_826 = None
    view_831: "f32[1024, 768]" = torch.ops.aten.view.default(div_110, [1024, 768]);  div_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_834: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_828, [1024, 1, 12, 64]);  view_828 = None
    permute_698: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_834, [1, 0, 2, 3]);  view_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_700: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_698, [0, 2, 1, 3]);  permute_698 = None
    view_836: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_700, [12, 1024, 64]);  permute_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_838: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_836, [12, 2, 512, 64]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_67: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_838, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_210: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_67, 4);  as_strided_67 = None
    permute_702: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_210, [0, 1, 4, 2, 3]);  unsqueeze_210 = None
    view_839: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_831, [1024, 1, 768]);  view_831 = None
    view_840: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_839, [1024, 1, 12, 64]);  view_839 = None
    permute_704: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_840, [1, 0, 2, 3]);  view_840 = None
    permute_705: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_704, [0, 2, 1, 3]);  permute_704 = None
    view_841: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_705, [12, 1024, 64]);  permute_705 = None
    view_842: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_841, [12, 2, 512, 64]);  view_841 = None
    as_strided_68: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_842, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_842 = None
    unsqueeze_211: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
    permute_706: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_211, [0, 1, 2, 4, 3]);  unsqueeze_211 = None
    permute_707: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_706, [0, 1, 2, 4, 3]);  permute_706 = None
    clone_55: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_843: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_55, [36, 512, 64]);  clone_55 = None
    permute_708: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_702, [0, 1, 4, 3, 2]);  permute_702 = None
    clone_56: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
    view_844: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_56, [36, 64, 512]);  clone_56 = None
    bmm_22: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_843, view_844)
    view_845: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, [12, 3, 512, 1, 512]);  bmm_22 = None
    permute_709: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_845, [0, 1, 2, 4, 3]);  view_845 = None
    view_846: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_709, [12, 3, 512, 512]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_44: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_846, [0, 0, 0, 1], 0.0);  view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_847: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_44, [12, 3, 512, 513]);  constant_pad_nd_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2542: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_847, 0, 0, 9223372036854775807);  view_847 = None
    slice_2543: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2542, 1, 0, 9223372036854775807)
    slice_2544: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2543, 2, 0, 256)
    slice_2545: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2544, 3, 0, 257);  slice_2544 = None
    copy_132: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_2545);  slice_8 = slice_2545 = None
    slice_scatter_484: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_132, 3, 256, 9223372036854775807);  slice_7 = copy_132 = None
    slice_scatter_485: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_6, slice_scatter_484, 2, 0, 9223372036854775807);  slice_6 = slice_scatter_484 = None
    slice_scatter_486: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, slice_scatter_485, 1, 0, -1);  slice_5 = slice_scatter_485 = None
    slice_scatter_487: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_486, 0, 0, 9223372036854775807);  full = slice_scatter_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    select_220: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2542, 1, -1)
    slice_2558: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_220, 1, 256, 9223372036854775807);  select_220 = None
    slice_2559: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2558, 2, 0, 257);  slice_2558 = None
    slice_2563: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_487, 0, 0, 9223372036854775807)
    select_222: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2563, 1, -1)
    slice_2564: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_222, 1, 0, 9223372036854775807)
    slice_2565: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2564, 2, 256, 9223372036854775807)
    copy_133: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2565, slice_2559);  slice_2565 = slice_2559 = None
    slice_scatter_488: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2564, copy_133, 2, 256, 9223372036854775807);  slice_2564 = copy_133 = None
    slice_scatter_489: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_222, slice_scatter_488, 1, 0, 9223372036854775807);  select_222 = slice_scatter_488 = None
    select_scatter_44: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2563, slice_scatter_489, 1, -1);  slice_2563 = slice_scatter_489 = None
    slice_scatter_490: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_487, select_scatter_44, 0, 0, 9223372036854775807);  slice_scatter_487 = select_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2573: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2543, 2, -257, -1);  slice_2543 = None
    slice_2574: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2573, 3, 257, 9223372036854775807);  slice_2573 = None
    slice_2579: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_490, 0, 0, 9223372036854775807)
    slice_2580: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2579, 1, 1, 9223372036854775807)
    slice_2581: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2580, 2, 0, 9223372036854775807)
    slice_2582: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2581, 3, 0, 256)
    copy_134: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2582, slice_2574);  slice_2582 = slice_2574 = None
    slice_scatter_491: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2581, copy_134, 3, 0, 256);  slice_2581 = copy_134 = None
    slice_scatter_492: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2580, slice_scatter_491, 2, 0, 9223372036854775807);  slice_2580 = slice_scatter_491 = None
    slice_scatter_493: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2579, slice_scatter_492, 1, 1, 9223372036854775807);  slice_2579 = slice_scatter_492 = None
    slice_scatter_494: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_490, slice_scatter_493, 0, 0, 9223372036854775807);  slice_scatter_490 = slice_scatter_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    select_225: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2542, 1, 0);  slice_2542 = None
    slice_2591: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_225, 1, 0, 255);  select_225 = None
    slice_2592: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2591, 2, -255, 9223372036854775807);  slice_2591 = None
    slice_2596: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_494, 0, 0, 9223372036854775807)
    select_227: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2596, 1, 0)
    slice_2597: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_227, 1, 1, 256)
    slice_2598: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2597, 2, 1, 256)
    copy_135: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2598, slice_2592);  slice_2598 = slice_2592 = None
    slice_scatter_495: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2597, copy_135, 2, 1, 256);  slice_2597 = copy_135 = None
    slice_scatter_496: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_227, slice_scatter_495, 1, 1, 256);  select_227 = slice_scatter_495 = None
    select_scatter_45: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2596, slice_scatter_496, 1, 0);  slice_2596 = slice_scatter_496 = None
    slice_scatter_497: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_494, select_scatter_45, 0, 0, 9223372036854775807);  slice_scatter_494 = select_scatter_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_850: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_497, [1, 12, 1024, 513]);  slice_scatter_497 = None
    permute_712: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_850, [0, 2, 1, 3]);  view_850 = None
    slice_2610: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_712, 0, 0, 9223372036854775807)
    slice_2611: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2610, 1, 0, 256)
    slice_2612: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2611, 2, 0, 9223372036854775807)
    slice_2613: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2612, 3, 0, 257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_89: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_2613);  convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    copy_136: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2613, where_89);  slice_2613 = where_89 = None
    slice_scatter_498: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2612, copy_136, 3, 0, 257);  slice_2612 = copy_136 = None
    slice_scatter_499: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2611, slice_scatter_498, 2, 0, 9223372036854775807);  slice_2611 = slice_scatter_498 = None
    slice_scatter_500: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2610, slice_scatter_499, 1, 0, 256);  slice_2610 = slice_scatter_499 = None
    slice_scatter_501: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_712, slice_scatter_500, 0, 0, 9223372036854775807);  permute_712 = slice_scatter_500 = None
    permute_715: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_501, [0, 2, 1, 3]);  slice_scatter_501 = None
    view_853: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_715, [12, 4, 256, 513]);  permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_855: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_853, [1, 12, 1024, 513]);  view_853 = None
    permute_717: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
    slice_2633: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_717, 0, 0, 9223372036854775807)
    slice_2634: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2633, 1, -256, 9223372036854775807)
    slice_2635: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2634, 2, 0, 9223372036854775807)
    slice_2636: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2635, 3, -257, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_90: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_2, slice_2636);  convert_element_type_1 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    copy_137: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2636, where_90);  slice_2636 = where_90 = None
    slice_scatter_502: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2635, copy_137, 3, -257, 9223372036854775807);  slice_2635 = copy_137 = None
    slice_scatter_503: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2634, slice_scatter_502, 2, 0, 9223372036854775807);  slice_2634 = slice_scatter_502 = None
    slice_scatter_504: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2633, slice_scatter_503, 1, -256, 9223372036854775807);  slice_2633 = slice_scatter_503 = None
    slice_scatter_505: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_717, slice_scatter_504, 0, 0, 9223372036854775807);  permute_717 = slice_scatter_504 = None
    permute_720: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_505, [0, 2, 1, 3]);  slice_scatter_505 = None
    view_858: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_720, [12, 4, 256, 513]);  permute_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_878: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_858, [1, 12, 1024, 513]);  view_858 = None
    permute_738: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
    add_123: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_738, permute_46);  permute_738 = permute_46 = None
    permute_740: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_123, [0, 2, 1, 3]);  add_123 = None
    view_881: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_740, [12, 4, 256, 513]);  permute_740 = None
    view_882: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_881, [1, 12, 1024, 513]);  view_881 = None
    permute_741: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_57: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
    amax_11: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_57, [-1], True)
    sub_92: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_57, amax_11);  clone_57 = amax_11 = None
    exp_11: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
    sum_12: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_117: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(div_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_95: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, div_117);  full_default_1 = div_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    native_dropout_33 = torch.ops.aten.native_dropout.default(where_95, 0.1, True);  where_95 = None
    getitem_110: "f32[1, 1024, 12, 513]" = native_dropout_33[0]
    getitem_111: "b8[1, 1024, 12, 513]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_883: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_830, [1024, 1, 12, 64]);  view_830 = None
    permute_742: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_883, [1, 0, 2, 3]);  view_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_743: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
    view_884: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_743, [12, 4, 256, 513]);  permute_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_744: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_742, [0, 2, 1, 3]);  permute_742 = None
    view_885: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_744, [12, 1024, 64]);  permute_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_46: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_885, [0, 0, 256, 256], -1.0);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_71: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_47: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_884, [0, 257], 0.0);  view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_886: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_47, [12, 4, -1]);  constant_pad_nd_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2766: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_886, 0, 0, 9223372036854775807);  view_886 = None
    slice_2767: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2766, 1, 0, 9223372036854775807);  slice_2766 = None
    slice_2768: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2767, 2, 0, -256);  slice_2767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_887: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2768, [12, 4, 256, 769]);  slice_2768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2769: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_887, 0, 0, 9223372036854775807);  view_887 = None
    slice_2770: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2769, 1, 0, 9223372036854775807);  slice_2769 = None
    slice_2771: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2770, 2, 0, 9223372036854775807);  slice_2770 = None
    slice_2772: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2771, 3, 0, -1);  slice_2771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_226: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2772, 4);  slice_2772 = None
    permute_745: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_226, [0, 1, 2, 4, 3]);  unsqueeze_226 = None
    unsqueeze_227: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
    permute_746: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_227, [0, 1, 4, 3, 2]);  unsqueeze_227 = None
    permute_747: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_745, [0, 1, 2, 4, 3]);  permute_745 = None
    view_888: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_747, [48, 256, 768]);  permute_747 = None
    permute_748: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_746, [0, 1, 4, 3, 2]);  permute_746 = None
    clone_58: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
    view_889: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_58, [48, 768, 64]);  clone_58 = None
    bmm_23: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_888, view_889)
    view_890: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_23, [12, 4, 256, 1, 64]);  bmm_23 = None
    permute_749: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_890, [0, 1, 2, 4, 3]);  view_890 = None
    view_891: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_749, [12, 4, 256, 64]);  permute_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_892: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_891, [1, 12, 1024, 64]);  view_891 = None
    permute_750: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_751: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_750, [1, 0, 2, 3]);  permute_750 = None
    clone_59: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
    view_893: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_59, [1024, 1, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_752: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_893, [1, 0, 2]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_894: "f32[1024, 768]" = torch.ops.aten.view.default(permute_752, [1024, 768]);  permute_752 = None
    permute_753: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_69: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_184, view_894, permute_753);  primals_184 = None
    view_895: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_69, [1, 1024, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    native_dropout_34 = torch.ops.aten.native_dropout.default(view_895, 0.1, True);  view_895 = None
    getitem_112: "f32[1, 1024, 768]" = native_dropout_34[0]
    getitem_113: "b8[1, 1024, 768]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_125: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_112, add_120);  getitem_112 = add_120 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_115: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_126: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_94: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_125, getitem_115);  add_125 = getitem_115 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = None
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_185)
    add_127: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_186);  mul_90 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_896: "f32[1024, 768]" = torch.ops.aten.view.default(add_127, [1024, 768])
    permute_754: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_70: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_188, view_896, permute_754);  primals_188 = None
    view_897: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_91: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, 0.5)
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, 0.7071067811865476);  view_897 = None
    erf_11: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_128: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_128);  mul_91 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_898: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_93, [1024, 3072]);  mul_93 = None
    permute_755: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_71: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_190, view_898, permute_755);  primals_190 = None
    view_899: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_71, [1, 1024, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_899, 0.1, True);  view_899 = None
    getitem_116: "f32[1, 1024, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 1024, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_127);  getitem_116 = add_127 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_130: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_95: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_119);  add_129 = getitem_119 = None
    mul_94: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = None
    mul_95: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, primals_191)
    add_131: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, primals_192);  mul_95 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1348, code: hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
    slice_2773: "f32[1, 1024, 768]" = torch.ops.aten.slice.Tensor(add_131, 0, 0, 9223372036854775807);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_120: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_756: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_755, [1, 0]);  permute_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_760: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_754, [1, 0]);  permute_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_121: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_764: "f32[768, 768]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_772: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_888, [0, 2, 1]);  view_888 = None
    permute_773: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_889, [0, 2, 1]);  view_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_12: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_783: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_843, [0, 2, 1]);  view_843 = None
    permute_784: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_844, [0, 2, 1]);  view_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_795: "f32[768, 768]" = torch.ops.aten.permute.default(permute_696, [1, 0]);  permute_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_799: "f32[768, 768]" = torch.ops.aten.permute.default(permute_695, [1, 0]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_808: "f32[768, 768]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_123: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_814: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_818: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_691, [1, 0]);  permute_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_124: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_822: "f32[768, 768]" = torch.ops.aten.permute.default(permute_690, [1, 0]);  permute_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_830: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_813, [0, 2, 1]);  view_813 = None
    permute_831: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_814, [0, 2, 1]);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_13: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_841: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_768, [0, 2, 1]);  view_768 = None
    permute_842: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_769, [0, 2, 1]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_853: "f32[768, 768]" = torch.ops.aten.permute.default(permute_633, [1, 0]);  permute_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_857: "f32[768, 768]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_866: "f32[768, 768]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_126: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_872: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_876: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_127: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_880: "f32[768, 768]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_888: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
    permute_889: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_14: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_899: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
    permute_900: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_694, [0, 2, 1]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_911: "f32[768, 768]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_915: "f32[768, 768]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_924: "f32[768, 768]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_129: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_930: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_934: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_130: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_938: "f32[768, 768]" = torch.ops.aten.permute.default(permute_564, [1, 0]);  permute_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_946: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None
    permute_947: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_664, [0, 2, 1]);  view_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_15: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_957: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None
    permute_958: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_619, [0, 2, 1]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_969: "f32[768, 768]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_973: "f32[768, 768]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_982: "f32[768, 768]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_132: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_988: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_992: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_133: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_996: "f32[768, 768]" = torch.ops.aten.permute.default(permute_501, [1, 0]);  permute_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1004: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None
    permute_1005: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_589, [0, 2, 1]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_16: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1015: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None
    permute_1016: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_544, [0, 2, 1]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1027: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1031: "f32[768, 768]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1040: "f32[768, 768]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_135: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1046: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1050: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_439, [1, 0]);  permute_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_136: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1054: "f32[768, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1062: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
    permute_1063: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_514, [0, 2, 1]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1073: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None
    permute_1074: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1085: "f32[768, 768]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1089: "f32[768, 768]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1098: "f32[768, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_138: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1104: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1108: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_139: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1112: "f32[768, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1120: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
    permute_1121: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_18: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1131: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None
    permute_1132: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1143: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1147: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1156: "f32[768, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_141: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1162: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1166: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_142: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1178: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    permute_1179: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_19: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1189: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    permute_1190: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1201: "f32[768, 768]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1205: "f32[768, 768]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_144: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1220: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1224: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_145: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1236: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    permute_1237: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1247: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    permute_1248: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_147: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1278: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1282: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_148: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1294: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    permute_1295: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_21: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1305: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    permute_1306: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1317: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1321: "f32[768, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_150: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1336: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1340: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_151: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1344: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1352: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
    permute_1353: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_22: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1363: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    permute_1364: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1375: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_153: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    permute_1394: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    permute_1398: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_154: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    permute_1402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    permute_1410: "f32[48, 768, 256]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    permute_1411: "f32[48, 64, 768]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    alias_23: "f32[1, 1024, 12, 513]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    permute_1421: "f32[36, 64, 512]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    permute_1422: "f32[36, 512, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    permute_1433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    permute_1437: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    permute_1446: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [slice_2773, primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, view, slice_64, rev_1, unsqueeze_16, getitem_1, view_69, getitem_3, mul_1, view_71, addmm_4, view_73, getitem_7, mul_6, view_75, getitem_11, view_144, getitem_13, mul_9, view_146, addmm_10, view_148, getitem_17, mul_14, view_150, getitem_21, view_219, getitem_23, mul_17, view_221, addmm_16, view_223, getitem_27, mul_22, view_225, getitem_31, view_294, getitem_33, mul_25, view_296, addmm_22, view_298, getitem_37, mul_30, view_300, getitem_41, view_369, getitem_43, mul_33, view_371, addmm_28, view_373, getitem_47, mul_38, view_375, getitem_51, view_444, getitem_53, mul_41, view_446, addmm_34, view_448, getitem_57, mul_46, view_450, getitem_61, view_519, getitem_63, mul_49, view_521, addmm_40, view_523, getitem_67, mul_54, view_525, getitem_71, view_594, getitem_73, mul_57, view_596, addmm_46, view_598, getitem_77, mul_62, view_600, getitem_81, view_669, getitem_83, mul_65, view_671, addmm_52, view_673, getitem_87, mul_70, view_675, getitem_91, view_744, getitem_93, mul_73, view_746, addmm_58, view_748, getitem_97, mul_78, view_750, getitem_101, view_819, getitem_103, mul_81, view_821, addmm_64, view_823, getitem_107, mul_86, view_825, getitem_111, view_894, getitem_113, mul_89, view_896, addmm_70, view_898, getitem_117, mul_94, div_120, permute_756, permute_760, div_121, permute_764, permute_772, permute_773, alias_12, permute_783, permute_784, permute_795, permute_799, permute_808, div_123, permute_814, permute_818, div_124, permute_822, permute_830, permute_831, alias_13, permute_841, permute_842, permute_853, permute_857, permute_866, div_126, permute_872, permute_876, div_127, permute_880, permute_888, permute_889, alias_14, permute_899, permute_900, permute_911, permute_915, permute_924, div_129, permute_930, permute_934, div_130, permute_938, permute_946, permute_947, alias_15, permute_957, permute_958, permute_969, permute_973, permute_982, div_132, permute_988, permute_992, div_133, permute_996, permute_1004, permute_1005, alias_16, permute_1015, permute_1016, permute_1027, permute_1031, permute_1040, div_135, permute_1046, permute_1050, div_136, permute_1054, permute_1062, permute_1063, alias_17, permute_1073, permute_1074, permute_1085, permute_1089, permute_1098, div_138, permute_1104, permute_1108, div_139, permute_1112, permute_1120, permute_1121, alias_18, permute_1131, permute_1132, permute_1143, permute_1147, permute_1156, div_141, permute_1162, permute_1166, div_142, permute_1170, permute_1178, permute_1179, alias_19, permute_1189, permute_1190, permute_1201, permute_1205, permute_1214, div_144, permute_1220, permute_1224, div_145, permute_1228, permute_1236, permute_1237, alias_20, permute_1247, permute_1248, permute_1259, permute_1263, permute_1272, div_147, permute_1278, permute_1282, div_148, permute_1286, permute_1294, permute_1295, alias_21, permute_1305, permute_1306, permute_1317, permute_1321, permute_1330, div_150, permute_1336, permute_1340, div_151, permute_1344, permute_1352, permute_1353, alias_22, permute_1363, permute_1364, permute_1375, permute_1379, permute_1388, div_153, permute_1394, permute_1398, div_154, permute_1402, permute_1410, permute_1411, alias_23, permute_1421, permute_1422, permute_1433, permute_1437, permute_1446]
    