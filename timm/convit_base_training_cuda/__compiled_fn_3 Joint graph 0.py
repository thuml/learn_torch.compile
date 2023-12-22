from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 196, 768]"; primals_2: "f32[1, 1, 768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[16]"; primals_6: "f32[768]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[16]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[16]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[16]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[16]"; primals_26: "f32[768]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[16]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[768]"; primals_35: "f32[16]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[16]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[16]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[16]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768, 3, 16, 16]"; primals_64: "f32[768]"; primals_65: "f32[1536, 768]"; primals_66: "f32[16, 3]"; primals_67: "f32[16]"; primals_68: "f32[768, 768]"; primals_69: "f32[768, 768]"; primals_70: "f32[768]"; primals_71: "f32[3072, 768]"; primals_72: "f32[3072]"; primals_73: "f32[768, 3072]"; primals_74: "f32[768]"; primals_75: "f32[1536, 768]"; primals_76: "f32[16, 3]"; primals_77: "f32[16]"; primals_78: "f32[768, 768]"; primals_79: "f32[768, 768]"; primals_80: "f32[768]"; primals_81: "f32[3072, 768]"; primals_82: "f32[3072]"; primals_83: "f32[768, 3072]"; primals_84: "f32[768]"; primals_85: "f32[1536, 768]"; primals_86: "f32[16, 3]"; primals_87: "f32[16]"; primals_88: "f32[768, 768]"; primals_89: "f32[768, 768]"; primals_90: "f32[768]"; primals_91: "f32[3072, 768]"; primals_92: "f32[3072]"; primals_93: "f32[768, 3072]"; primals_94: "f32[768]"; primals_95: "f32[1536, 768]"; primals_96: "f32[16, 3]"; primals_97: "f32[16]"; primals_98: "f32[768, 768]"; primals_99: "f32[768, 768]"; primals_100: "f32[768]"; primals_101: "f32[3072, 768]"; primals_102: "f32[3072]"; primals_103: "f32[768, 3072]"; primals_104: "f32[768]"; primals_105: "f32[1536, 768]"; primals_106: "f32[16, 3]"; primals_107: "f32[16]"; primals_108: "f32[768, 768]"; primals_109: "f32[768, 768]"; primals_110: "f32[768]"; primals_111: "f32[3072, 768]"; primals_112: "f32[3072]"; primals_113: "f32[768, 3072]"; primals_114: "f32[768]"; primals_115: "f32[1536, 768]"; primals_116: "f32[16, 3]"; primals_117: "f32[16]"; primals_118: "f32[768, 768]"; primals_119: "f32[768, 768]"; primals_120: "f32[768]"; primals_121: "f32[3072, 768]"; primals_122: "f32[3072]"; primals_123: "f32[768, 3072]"; primals_124: "f32[768]"; primals_125: "f32[1536, 768]"; primals_126: "f32[16, 3]"; primals_127: "f32[16]"; primals_128: "f32[768, 768]"; primals_129: "f32[768, 768]"; primals_130: "f32[768]"; primals_131: "f32[3072, 768]"; primals_132: "f32[3072]"; primals_133: "f32[768, 3072]"; primals_134: "f32[768]"; primals_135: "f32[1536, 768]"; primals_136: "f32[16, 3]"; primals_137: "f32[16]"; primals_138: "f32[768, 768]"; primals_139: "f32[768, 768]"; primals_140: "f32[768]"; primals_141: "f32[3072, 768]"; primals_142: "f32[3072]"; primals_143: "f32[768, 3072]"; primals_144: "f32[768]"; primals_145: "f32[1536, 768]"; primals_146: "f32[16, 3]"; primals_147: "f32[16]"; primals_148: "f32[768, 768]"; primals_149: "f32[768, 768]"; primals_150: "f32[768]"; primals_151: "f32[3072, 768]"; primals_152: "f32[3072]"; primals_153: "f32[768, 3072]"; primals_154: "f32[768]"; primals_155: "f32[1536, 768]"; primals_156: "f32[16, 3]"; primals_157: "f32[16]"; primals_158: "f32[768, 768]"; primals_159: "f32[768, 768]"; primals_160: "f32[768]"; primals_161: "f32[3072, 768]"; primals_162: "f32[3072]"; primals_163: "f32[768, 3072]"; primals_164: "f32[768]"; primals_165: "f32[2304, 768]"; primals_166: "f32[768, 768]"; primals_167: "f32[768]"; primals_168: "f32[3072, 768]"; primals_169: "f32[3072]"; primals_170: "f32[768, 3072]"; primals_171: "f32[768]"; primals_172: "f32[2304, 768]"; primals_173: "f32[768, 768]"; primals_174: "f32[768]"; primals_175: "f32[3072, 768]"; primals_176: "f32[3072]"; primals_177: "f32[768, 3072]"; primals_178: "f32[768]"; primals_179: "f32[1000, 768]"; primals_180: "f32[1000]"; primals_181: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_181, primals_63, primals_64, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    add: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(permute, primals_1);  permute = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:363, code: x = self.pos_drop(x)
    clone: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 196, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = None
    mul: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_2: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_1: "i64[1, 14]" = torch.ops.aten.view.default(iota, [1, -1]);  iota = None
    iota_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_2: "i64[14, 1]" = torch.ops.aten.view.default(iota_1, [-1, 1]);  iota_1 = None
    sub_1: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_1, view_2);  view_1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_1, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_1, 1);  sub_1 = None
    expand_1: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze, [14, 14, 14]);  unsqueeze = None
    clone_2: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_3: "i64[196, 14]" = torch.ops.aten.view.default(clone_2, [196, 14]);  clone_2 = None
    unsqueeze_1: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_3, 2);  view_3 = None
    expand_2: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_1, [196, 14, 14]);  unsqueeze_1 = None
    clone_3: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_4: "i64[196, 196]" = torch.ops.aten.view.default(clone_3, [196, 196]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_1: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat, 2)
    pow_2: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 2)
    add_3: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_2: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
    slice_1: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_2: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
    select: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_3, 3, 2);  slice_3 = None
    copy: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select, unsqueeze_2);  select = unsqueeze_2 = None
    slice_4: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_5: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807)
    slice_6: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 9223372036854775807)
    select_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_6, copy, 3, 2);  slice_6 = copy = None
    slice_scatter: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_5, select_scatter, 2, 0, 9223372036854775807);  slice_5 = select_scatter = None
    slice_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_4, slice_scatter, 1, 0, 9223372036854775807);  slice_4 = slice_scatter = None
    slice_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_1, 0, 0, 9223372036854775807);  full = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_3: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
    slice_13: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
    slice_14: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807);  slice_14 = None
    select_3: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_15, 3, 1);  slice_15 = None
    copy_1: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_3, unsqueeze_3);  select_3 = unsqueeze_3 = None
    slice_16: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
    slice_17: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 9223372036854775807)
    slice_18: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 9223372036854775807)
    select_scatter_1: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_18, copy_1, 3, 1);  slice_18 = copy_1 = None
    slice_scatter_3: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_17, select_scatter_1, 2, 0, 9223372036854775807);  slice_17 = select_scatter_1 = None
    slice_scatter_4: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_16, slice_scatter_3, 1, 0, 9223372036854775807);  slice_16 = slice_scatter_3 = None
    slice_scatter_5: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_2, slice_scatter_4, 0, 0, 9223372036854775807);  slice_scatter_2 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_4: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat, 0);  repeat = None
    slice_25: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 9223372036854775807)
    slice_26: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807);  slice_26 = None
    select_6: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_27, 3, 0);  slice_27 = None
    copy_2: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_6, unsqueeze_4);  select_6 = unsqueeze_4 = None
    slice_28: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 9223372036854775807)
    slice_29: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 9223372036854775807)
    slice_30: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_29, 2, 0, 9223372036854775807)
    select_scatter_2: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_30, copy_2, 3, 0);  slice_30 = copy_2 = None
    slice_scatter_6: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_29, select_scatter_2, 2, 0, 9223372036854775807);  slice_29 = select_scatter_2 = None
    slice_scatter_7: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_28, slice_scatter_6, 1, 0, 9223372036854775807);  slice_28 = slice_scatter_6 = None
    slice_scatter_8: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_5 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_8, device(type='cuda', index=0));  slice_scatter_8 = None
    convert_element_type: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(add_2, [1568, 768])
    mm: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_5, permute_1)
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm, [8, 196, 1536]);  mm = None
    view_7: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_6, [8, 196, 2, 16, 48]);  view_6 = None
    permute_2: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_7, [2, 0, 3, 1, 4]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_8: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    select_9: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_2, 0, 1);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_3: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_3: "f32[3, 16]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    clone_4: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_8: "f32[307328, 3]" = torch.ops.aten.view.default(clone_4, [307328, 3]);  clone_4 = None
    mm_1: "f32[307328, 16]" = torch.ops.aten.mm.default(view_8, permute_3);  permute_3 = None
    view_9: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_1, [8, 196, 196, 16]);  mm_1 = None
    add_4: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_9, primals_67);  view_9 = primals_67 = None
    permute_4: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_5: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_9, [0, 1, 3, 2]);  select_9 = None
    expand_4: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 196, 48]);  select_8 = None
    clone_5: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_5, [128, 196, 48]);  clone_5 = None
    expand_5: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_5, [8, 16, 48, 196]);  permute_5 = None
    clone_6: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_11: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_6, [128, 48, 196]);  clone_6 = None
    bmm: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 16, 196, 196]);  bmm = None
    mul_2: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_12, 0.14433756729740643);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_2, [-1], True)
    sub_2: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_2, amax);  mul_2 = amax = None
    exp: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_7: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    amax_1: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_7, [-1], True)
    sub_3: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_7, amax_1);  clone_7 = amax_1 = None
    exp_1: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_13: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_5, [1, -1, 1, 1]);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13)
    alias_2: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid)
    sub_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid);  sigmoid = None
    mul_3: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_4, div)
    sigmoid_1: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13);  view_13 = None
    alias_3: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_1)
    mul_4: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_1, div_1)
    add_5: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_3: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_5, [-1])
    unsqueeze_5: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_3, -1);  sum_3 = None
    clone_8: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_5)
    div_2: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_5, unsqueeze_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_9: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    view_14: "f32[1568, 768]" = torch.ops.aten.view.default(add_2, [1568, 768]);  add_2 = None
    mm_2: "f32[1568, 768]" = torch.ops.aten.mm.default(view_14, permute_6)
    view_15: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_2, [8, 196, 768]);  mm_2 = None
    view_16: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_15, [8, 196, 16, 48]);  view_15 = None
    permute_7: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_6: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_9, [8, 16, 196, 196]);  clone_9 = None
    view_17: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_6, [128, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_7, [8, 16, 196, 48]);  permute_7 = None
    clone_10: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_18: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_10, [128, 196, 48]);  clone_10 = None
    bmm_1: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_17, view_18)
    view_19: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_1, [8, 16, 196, 48]);  bmm_1 = None
    permute_8: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
    clone_11: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_20: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_11, [8, 196, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_21: "f32[1568, 768]" = torch.ops.aten.view.default(view_20, [1568, 768]);  view_20 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_70, view_21, permute_9);  primals_70 = None
    view_22: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm, [8, 196, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_6: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(clone, clone_12);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_13: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_13, getitem_3);  clone_13 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_6);  mul_5 = None
    add_8: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_7);  mul_6 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(add_8, [1568, 768]);  add_8 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_1: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_72, view_23, permute_10);  primals_72 = None
    view_24: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 196, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.5)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476)
    erf: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_25: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_14, [1568, 3072]);  clone_14 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_2: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_74, view_25, permute_11);  primals_74 = None
    view_26: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_2, [8, 196, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_10: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_6, clone_15);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_16, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_6: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_16, getitem_5);  clone_16 = None
    mul_10: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
    mul_11: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_8);  mul_10 = None
    add_12: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_9);  mul_11 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_1: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_27: "i64[1, 14]" = torch.ops.aten.view.default(iota_2, [1, -1]);  iota_2 = None
    iota_3: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_28: "i64[14, 1]" = torch.ops.aten.view.default(iota_3, [-1, 1]);  iota_3 = None
    sub_7: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_27, view_28);  view_27 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_1: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_7, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_6: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_7, 1);  sub_7 = None
    expand_8: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_6, [14, 14, 14]);  unsqueeze_6 = None
    clone_17: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_29: "i64[196, 14]" = torch.ops.aten.view.default(clone_17, [196, 14]);  clone_17 = None
    unsqueeze_7: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_29, 2);  view_29 = None
    expand_9: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_7, [196, 14, 14]);  unsqueeze_7 = None
    clone_18: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_30: "i64[196, 196]" = torch.ops.aten.view.default(clone_18, [196, 196]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_3: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_1, 2)
    pow_4: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_30, 2)
    add_13: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_3, pow_4);  pow_3 = pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_8: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_13, 0);  add_13 = None
    slice_34: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_35: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_34, 1, 0, 9223372036854775807);  slice_34 = None
    slice_36: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_35, 2, 0, 9223372036854775807);  slice_35 = None
    select_10: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_36, 3, 2);  slice_36 = None
    copy_3: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_10, unsqueeze_8);  select_10 = unsqueeze_8 = None
    slice_37: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_38: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_37, 1, 0, 9223372036854775807)
    slice_39: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_38, 2, 0, 9223372036854775807)
    select_scatter_3: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_39, copy_3, 3, 2);  slice_39 = copy_3 = None
    slice_scatter_9: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_38, select_scatter_3, 2, 0, 9223372036854775807);  slice_38 = select_scatter_3 = None
    slice_scatter_10: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_37, slice_scatter_9, 1, 0, 9223372036854775807);  slice_37 = slice_scatter_9 = None
    slice_scatter_11: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_10, 0, 0, 9223372036854775807);  full_1 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_9: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    slice_46: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 9223372036854775807)
    slice_47: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_46, 1, 0, 9223372036854775807);  slice_46 = None
    slice_48: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_47, 2, 0, 9223372036854775807);  slice_47 = None
    select_13: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_48, 3, 1);  slice_48 = None
    copy_4: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_13, unsqueeze_9);  select_13 = unsqueeze_9 = None
    slice_49: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 9223372036854775807)
    slice_50: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_49, 1, 0, 9223372036854775807)
    slice_51: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_50, 2, 0, 9223372036854775807)
    select_scatter_4: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_51, copy_4, 3, 1);  slice_51 = copy_4 = None
    slice_scatter_12: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_50, select_scatter_4, 2, 0, 9223372036854775807);  slice_50 = select_scatter_4 = None
    slice_scatter_13: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_49, slice_scatter_12, 1, 0, 9223372036854775807);  slice_49 = slice_scatter_12 = None
    slice_scatter_14: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_11 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_10: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_1, 0);  repeat_1 = None
    slice_58: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, 9223372036854775807)
    slice_59: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_58, 1, 0, 9223372036854775807);  slice_58 = None
    slice_60: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_59, 2, 0, 9223372036854775807);  slice_59 = None
    select_16: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_60, 3, 0);  slice_60 = None
    copy_5: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_16, unsqueeze_10);  select_16 = unsqueeze_10 = None
    slice_61: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, 9223372036854775807)
    slice_62: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_61, 1, 0, 9223372036854775807)
    slice_63: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_62, 2, 0, 9223372036854775807)
    select_scatter_5: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_63, copy_5, 3, 0);  slice_63 = copy_5 = None
    slice_scatter_15: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_62, select_scatter_5, 2, 0, 9223372036854775807);  slice_62 = select_scatter_5 = None
    slice_scatter_16: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_61, slice_scatter_15, 1, 0, 9223372036854775807);  slice_61 = slice_scatter_15 = None
    slice_scatter_17: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_14, slice_scatter_16, 0, 0, 9223372036854775807);  slice_scatter_14 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_1: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_17, device(type='cuda', index=0));  slice_scatter_17 = None
    convert_element_type_1: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_1, torch.float32);  device_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_12: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    view_31: "f32[1568, 768]" = torch.ops.aten.view.default(add_12, [1568, 768])
    mm_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_31, permute_12)
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_3, [8, 196, 1536]);  mm_3 = None
    view_33: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_32, [8, 196, 2, 16, 48]);  view_32 = None
    permute_13: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_18: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 0)
    select_19: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_13, 0, 1);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_10: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_1, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_14: "f32[3, 16]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    clone_19: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_34: "f32[307328, 3]" = torch.ops.aten.view.default(clone_19, [307328, 3]);  clone_19 = None
    mm_4: "f32[307328, 16]" = torch.ops.aten.mm.default(view_34, permute_14);  permute_14 = None
    view_35: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_4, [8, 196, 196, 16]);  mm_4 = None
    add_14: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_35, primals_77);  view_35 = primals_77 = None
    permute_15: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_14, [0, 3, 1, 2]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_11: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_18, [8, 16, 196, 48]);  select_18 = None
    clone_20: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_36: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_20, [128, 196, 48]);  clone_20 = None
    expand_12: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 196]);  permute_16 = None
    clone_21: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_37: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_21, [128, 48, 196]);  clone_21 = None
    bmm_2: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 16, 196, 196]);  bmm_2 = None
    mul_12: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_38, 0.14433756729740643);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_2: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_12, [-1], True)
    sub_8: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_12, amax_2);  mul_12 = amax_2 = None
    exp_2: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_4: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_3: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_4);  exp_2 = sum_4 = None
    alias_4: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_22: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    amax_3: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_22, [-1], True)
    sub_9: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_22, amax_3);  clone_22 = amax_3 = None
    exp_3: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_4: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_5);  exp_3 = sum_5 = None
    alias_5: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_39: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_10, [1, -1, 1, 1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39)
    alias_6: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_2)
    sub_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2);  sigmoid_2 = None
    mul_13: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_10, div_3)
    sigmoid_3: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39);  view_39 = None
    alias_7: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    mul_14: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_3, div_4)
    add_15: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_6: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_15, [-1])
    unsqueeze_11: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_6, -1);  sum_6 = None
    clone_23: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_15)
    div_5: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_15, unsqueeze_11);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_24: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    view_40: "f32[1568, 768]" = torch.ops.aten.view.default(add_12, [1568, 768]);  add_12 = None
    mm_5: "f32[1568, 768]" = torch.ops.aten.mm.default(view_40, permute_17)
    view_41: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_5, [8, 196, 768]);  mm_5 = None
    view_42: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_41, [8, 196, 16, 48]);  view_41 = None
    permute_18: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_13: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_24, [8, 16, 196, 196]);  clone_24 = None
    view_43: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_13, [128, 196, 196]);  expand_13 = None
    expand_14: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_18, [8, 16, 196, 48]);  permute_18 = None
    clone_25: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_44: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_25, [128, 196, 48]);  clone_25 = None
    bmm_3: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_3, [8, 16, 196, 48]);  bmm_3 = None
    permute_19: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_26: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_46: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_26, [8, 196, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(view_46, [1568, 768]);  view_46 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_3: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_80, view_47, permute_20);  primals_80 = None
    view_48: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_3, [8, 196, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_27: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_16: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_10, clone_27);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_28: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_28, getitem_7);  clone_28 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = None
    mul_16: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_11);  mul_15 = None
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_12);  mul_16 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_49: "f32[1568, 768]" = torch.ops.aten.view.default(add_18, [1568, 768]);  add_18 = None
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_4: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_82, view_49, permute_21);  primals_82 = None
    view_50: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 196, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_19);  mul_17 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_51: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_29, [1568, 3072]);  clone_29 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_5: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_84, view_51, permute_22);  primals_84 = None
    view_52: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_5, [8, 196, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_20: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_16, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_31: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_31, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_12: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_31, getitem_9);  clone_31 = None
    mul_20: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_4);  sub_12 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_13);  mul_20 = None
    add_22: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_14);  mul_21 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_2: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_53: "i64[1, 14]" = torch.ops.aten.view.default(iota_4, [1, -1]);  iota_4 = None
    iota_5: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_54: "i64[14, 1]" = torch.ops.aten.view.default(iota_5, [-1, 1]);  iota_5 = None
    sub_13: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_53, view_54);  view_53 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_2: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_13, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_12: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_13, 1);  sub_13 = None
    expand_15: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_12, [14, 14, 14]);  unsqueeze_12 = None
    clone_32: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_55: "i64[196, 14]" = torch.ops.aten.view.default(clone_32, [196, 14]);  clone_32 = None
    unsqueeze_13: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_55, 2);  view_55 = None
    expand_16: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_13, [196, 14, 14]);  unsqueeze_13 = None
    clone_33: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_56: "i64[196, 196]" = torch.ops.aten.view.default(clone_33, [196, 196]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_5: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_2, 2)
    pow_6: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_56, 2)
    add_23: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_5, pow_6);  pow_5 = pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_14: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
    slice_67: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_68: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_67, 1, 0, 9223372036854775807);  slice_67 = None
    slice_69: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_68, 2, 0, 9223372036854775807);  slice_68 = None
    select_20: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_69, 3, 2);  slice_69 = None
    copy_6: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_20, unsqueeze_14);  select_20 = unsqueeze_14 = None
    slice_70: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_71: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_70, 1, 0, 9223372036854775807)
    slice_72: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_71, 2, 0, 9223372036854775807)
    select_scatter_6: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_72, copy_6, 3, 2);  slice_72 = copy_6 = None
    slice_scatter_18: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_71, select_scatter_6, 2, 0, 9223372036854775807);  slice_71 = select_scatter_6 = None
    slice_scatter_19: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_70, slice_scatter_18, 1, 0, 9223372036854775807);  slice_70 = slice_scatter_18 = None
    slice_scatter_20: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_19, 0, 0, 9223372036854775807);  full_2 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_15: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_56, 0);  view_56 = None
    slice_79: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 0, 9223372036854775807)
    slice_80: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_79, 1, 0, 9223372036854775807);  slice_79 = None
    slice_81: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_80, 2, 0, 9223372036854775807);  slice_80 = None
    select_23: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_81, 3, 1);  slice_81 = None
    copy_7: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_23, unsqueeze_15);  select_23 = unsqueeze_15 = None
    slice_82: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 0, 9223372036854775807)
    slice_83: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_82, 1, 0, 9223372036854775807)
    slice_84: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_83, 2, 0, 9223372036854775807)
    select_scatter_7: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_84, copy_7, 3, 1);  slice_84 = copy_7 = None
    slice_scatter_21: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_83, select_scatter_7, 2, 0, 9223372036854775807);  slice_83 = select_scatter_7 = None
    slice_scatter_22: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_82, slice_scatter_21, 1, 0, 9223372036854775807);  slice_82 = slice_scatter_21 = None
    slice_scatter_23: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_20, slice_scatter_22, 0, 0, 9223372036854775807);  slice_scatter_20 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_16: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_2, 0);  repeat_2 = None
    slice_91: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 9223372036854775807)
    slice_92: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_91, 1, 0, 9223372036854775807);  slice_91 = None
    slice_93: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_92, 2, 0, 9223372036854775807);  slice_92 = None
    select_26: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_93, 3, 0);  slice_93 = None
    copy_8: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_26, unsqueeze_16);  select_26 = unsqueeze_16 = None
    slice_94: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 9223372036854775807)
    slice_95: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_94, 1, 0, 9223372036854775807)
    slice_96: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_95, 2, 0, 9223372036854775807)
    select_scatter_8: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_96, copy_8, 3, 0);  slice_96 = copy_8 = None
    slice_scatter_24: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_95, select_scatter_8, 2, 0, 9223372036854775807);  slice_95 = select_scatter_8 = None
    slice_scatter_25: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_94, slice_scatter_24, 1, 0, 9223372036854775807);  slice_94 = slice_scatter_24 = None
    slice_scatter_26: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_23, slice_scatter_25, 0, 0, 9223372036854775807);  slice_scatter_23 = slice_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_2: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_26, device(type='cuda', index=0));  slice_scatter_26 = None
    convert_element_type_2: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_2, torch.float32);  device_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_23: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    view_57: "f32[1568, 768]" = torch.ops.aten.view.default(add_22, [1568, 768])
    mm_6: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_57, permute_23)
    view_58: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_6, [8, 196, 1536]);  mm_6 = None
    view_59: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_58, [8, 196, 2, 16, 48]);  view_58 = None
    permute_24: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_59, [2, 0, 3, 1, 4]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_28: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 0)
    select_29: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_24, 0, 1);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_17: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_2, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_25: "f32[3, 16]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    clone_34: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_60: "f32[307328, 3]" = torch.ops.aten.view.default(clone_34, [307328, 3]);  clone_34 = None
    mm_7: "f32[307328, 16]" = torch.ops.aten.mm.default(view_60, permute_25);  permute_25 = None
    view_61: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_7, [8, 196, 196, 16]);  mm_7 = None
    add_24: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_61, primals_87);  view_61 = primals_87 = None
    permute_26: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_27: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_29, [0, 1, 3, 2]);  select_29 = None
    expand_18: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_28, [8, 16, 196, 48]);  select_28 = None
    clone_35: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_62: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_35, [128, 196, 48]);  clone_35 = None
    expand_19: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_27, [8, 16, 48, 196]);  permute_27 = None
    clone_36: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_63: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_36, [128, 48, 196]);  clone_36 = None
    bmm_4: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 16, 196, 196]);  bmm_4 = None
    mul_22: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_64, 0.14433756729740643);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_4: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_22, [-1], True)
    sub_14: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_22, amax_4);  mul_22 = amax_4 = None
    exp_4: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_7);  exp_4 = sum_7 = None
    alias_8: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_37: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    amax_5: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_37, [-1], True)
    sub_15: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_37, amax_5);  clone_37 = amax_5 = None
    exp_5: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_8);  exp_5 = sum_8 = None
    alias_9: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_65: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_15, [1, -1, 1, 1]);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65)
    alias_10: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_4)
    sub_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4);  sigmoid_4 = None
    mul_23: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_16, div_6)
    sigmoid_5: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65);  view_65 = None
    alias_11: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    mul_24: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_5, div_7)
    add_25: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_9: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_25, [-1])
    unsqueeze_17: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_9, -1);  sum_9 = None
    clone_38: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_25)
    div_8: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_25, unsqueeze_17);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_39: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    view_66: "f32[1568, 768]" = torch.ops.aten.view.default(add_22, [1568, 768]);  add_22 = None
    mm_8: "f32[1568, 768]" = torch.ops.aten.mm.default(view_66, permute_28)
    view_67: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_8, [8, 196, 768]);  mm_8 = None
    view_68: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_67, [8, 196, 16, 48]);  view_67 = None
    permute_29: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_20: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_39, [8, 16, 196, 196]);  clone_39 = None
    view_69: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_20, [128, 196, 196]);  expand_20 = None
    expand_21: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_29, [8, 16, 196, 48]);  permute_29 = None
    clone_40: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_70: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_40, [128, 196, 48]);  clone_40 = None
    bmm_5: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_69, view_70)
    view_71: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_5, [8, 16, 196, 48]);  bmm_5 = None
    permute_30: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_41: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_72: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_41, [8, 196, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_73: "f32[1568, 768]" = torch.ops.aten.view.default(view_72, [1568, 768]);  view_72 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_6: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_90, view_73, permute_31);  primals_90 = None
    view_74: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_6, [8, 196, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_26: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_20, clone_42);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_11);  clone_43 = None
    mul_25: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_5);  sub_17 = None
    mul_26: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_16);  mul_25 = None
    add_28: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_17);  mul_26 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 768]" = torch.ops.aten.view.default(add_28, [1568, 768]);  add_28 = None
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_7: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_92, view_75, permute_32);  primals_92 = None
    view_76: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 196, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_2: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_29: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_29);  mul_27 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_44, [1568, 3072]);  clone_44 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_8: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_94, view_77, permute_33);  primals_94 = None
    view_78: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_8, [8, 196, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_30: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_26, clone_45);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_46: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_18: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_46, getitem_13);  clone_46 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_6);  sub_18 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_18);  mul_30 = None
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_19);  mul_31 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_3: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_79: "i64[1, 14]" = torch.ops.aten.view.default(iota_6, [1, -1]);  iota_6 = None
    iota_7: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_80: "i64[14, 1]" = torch.ops.aten.view.default(iota_7, [-1, 1]);  iota_7 = None
    sub_19: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_79, view_80);  view_79 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_3: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_19, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_18: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_19, 1);  sub_19 = None
    expand_22: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_18, [14, 14, 14]);  unsqueeze_18 = None
    clone_47: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_81: "i64[196, 14]" = torch.ops.aten.view.default(clone_47, [196, 14]);  clone_47 = None
    unsqueeze_19: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_81, 2);  view_81 = None
    expand_23: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_19, [196, 14, 14]);  unsqueeze_19 = None
    clone_48: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_82: "i64[196, 196]" = torch.ops.aten.view.default(clone_48, [196, 196]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_7: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_3, 2)
    pow_8: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 2)
    add_33: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_7, pow_8);  pow_7 = pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_20: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_33, 0);  add_33 = None
    slice_100: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807)
    slice_101: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_100, 1, 0, 9223372036854775807);  slice_100 = None
    slice_102: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_101, 2, 0, 9223372036854775807);  slice_101 = None
    select_30: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_102, 3, 2);  slice_102 = None
    copy_9: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_30, unsqueeze_20);  select_30 = unsqueeze_20 = None
    slice_103: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807)
    slice_104: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_103, 1, 0, 9223372036854775807)
    slice_105: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_104, 2, 0, 9223372036854775807)
    select_scatter_9: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_105, copy_9, 3, 2);  slice_105 = copy_9 = None
    slice_scatter_27: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_104, select_scatter_9, 2, 0, 9223372036854775807);  slice_104 = select_scatter_9 = None
    slice_scatter_28: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_103, slice_scatter_27, 1, 0, 9223372036854775807);  slice_103 = slice_scatter_27 = None
    slice_scatter_29: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_3, slice_scatter_28, 0, 0, 9223372036854775807);  full_3 = slice_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_21: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_82, 0);  view_82 = None
    slice_112: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 9223372036854775807)
    slice_113: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_112, 1, 0, 9223372036854775807);  slice_112 = None
    slice_114: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_113, 2, 0, 9223372036854775807);  slice_113 = None
    select_33: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_114, 3, 1);  slice_114 = None
    copy_10: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_33, unsqueeze_21);  select_33 = unsqueeze_21 = None
    slice_115: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 9223372036854775807)
    slice_116: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_115, 1, 0, 9223372036854775807)
    slice_117: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_116, 2, 0, 9223372036854775807)
    select_scatter_10: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_117, copy_10, 3, 1);  slice_117 = copy_10 = None
    slice_scatter_30: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_116, select_scatter_10, 2, 0, 9223372036854775807);  slice_116 = select_scatter_10 = None
    slice_scatter_31: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_115, slice_scatter_30, 1, 0, 9223372036854775807);  slice_115 = slice_scatter_30 = None
    slice_scatter_32: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_29, slice_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_29 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_22: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_3, 0);  repeat_3 = None
    slice_124: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    slice_125: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_124, 1, 0, 9223372036854775807);  slice_124 = None
    slice_126: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_125, 2, 0, 9223372036854775807);  slice_125 = None
    select_36: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_126, 3, 0);  slice_126 = None
    copy_11: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_36, unsqueeze_22);  select_36 = unsqueeze_22 = None
    slice_127: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    slice_128: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_127, 1, 0, 9223372036854775807)
    slice_129: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_128, 2, 0, 9223372036854775807)
    select_scatter_11: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_129, copy_11, 3, 0);  slice_129 = copy_11 = None
    slice_scatter_33: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_128, select_scatter_11, 2, 0, 9223372036854775807);  slice_128 = select_scatter_11 = None
    slice_scatter_34: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_127, slice_scatter_33, 1, 0, 9223372036854775807);  slice_127 = slice_scatter_33 = None
    slice_scatter_35: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, slice_scatter_34, 0, 0, 9223372036854775807);  slice_scatter_32 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_3: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_35, device(type='cuda', index=0));  slice_scatter_35 = None
    convert_element_type_3: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_3, torch.float32);  device_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_34: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    view_83: "f32[1568, 768]" = torch.ops.aten.view.default(add_32, [1568, 768])
    mm_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_83, permute_34)
    view_84: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_9, [8, 196, 1536]);  mm_9 = None
    view_85: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_84, [8, 196, 2, 16, 48]);  view_84 = None
    permute_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_85, [2, 0, 3, 1, 4]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_38: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 0)
    select_39: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_35, 0, 1);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_24: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_3, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_36: "f32[3, 16]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    clone_49: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_86: "f32[307328, 3]" = torch.ops.aten.view.default(clone_49, [307328, 3]);  clone_49 = None
    mm_10: "f32[307328, 16]" = torch.ops.aten.mm.default(view_86, permute_36);  permute_36 = None
    view_87: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_10, [8, 196, 196, 16]);  mm_10 = None
    add_34: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_87, primals_97);  view_87 = primals_97 = None
    permute_37: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_38: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_39, [0, 1, 3, 2]);  select_39 = None
    expand_25: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 196, 48]);  select_38 = None
    clone_50: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_88: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_50, [128, 196, 48]);  clone_50 = None
    expand_26: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_38, [8, 16, 48, 196]);  permute_38 = None
    clone_51: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_89: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_51, [128, 48, 196]);  clone_51 = None
    bmm_6: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_88, view_89)
    view_90: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 16, 196, 196]);  bmm_6 = None
    mul_32: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_90, 0.14433756729740643);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_6: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_32, [-1], True)
    sub_20: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_32, amax_6);  mul_32 = amax_6 = None
    exp_6: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_10: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_9: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_10);  exp_6 = sum_10 = None
    alias_12: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_52: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    amax_7: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_52, [-1], True)
    sub_21: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_52, amax_7);  clone_52 = amax_7 = None
    exp_7: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_10: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_11);  exp_7 = sum_11 = None
    alias_13: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_91: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_20, [1, -1, 1, 1]);  primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91)
    alias_14: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_6)
    sub_22: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_6);  sigmoid_6 = None
    mul_33: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_22, div_9)
    sigmoid_7: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91);  view_91 = None
    alias_15: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    mul_34: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_7, div_10)
    add_35: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_12: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_35, [-1])
    unsqueeze_23: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_12, -1);  sum_12 = None
    clone_53: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_35)
    div_11: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_35, unsqueeze_23);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_54: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_92: "f32[1568, 768]" = torch.ops.aten.view.default(add_32, [1568, 768]);  add_32 = None
    mm_11: "f32[1568, 768]" = torch.ops.aten.mm.default(view_92, permute_39)
    view_93: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_11, [8, 196, 768]);  mm_11 = None
    view_94: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_93, [8, 196, 16, 48]);  view_93 = None
    permute_40: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_27: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_54, [8, 16, 196, 196]);  clone_54 = None
    view_95: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_27, [128, 196, 196]);  expand_27 = None
    expand_28: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_40, [8, 16, 196, 48]);  permute_40 = None
    clone_55: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_96: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_55, [128, 196, 48]);  clone_55 = None
    bmm_7: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_95, view_96)
    view_97: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_7, [8, 16, 196, 48]);  bmm_7 = None
    permute_41: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_56: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_98: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_56, [8, 196, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_99: "f32[1568, 768]" = torch.ops.aten.view.default(view_98, [1568, 768]);  view_98 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_9: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_100, view_99, permute_42);  primals_100 = None
    view_100: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_9, [8, 196, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_36: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_30, clone_57);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_58: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_58, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_58, getitem_15);  clone_58 = None
    mul_35: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_7);  sub_23 = None
    mul_36: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_21);  mul_35 = None
    add_38: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_22);  mul_36 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.view.default(add_38, [1568, 768]);  add_38 = None
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_10: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_102, view_101, permute_43);  primals_102 = None
    view_102: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 196, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.5)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_39);  mul_37 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_59: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_59, [1568, 3072]);  clone_59 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_11: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_104, view_103, permute_44);  primals_104 = None
    view_104: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_11, [8, 196, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_60: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_40: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_36, clone_60);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_61: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_61, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_24: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_61, getitem_17);  clone_61 = None
    mul_40: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_8);  sub_24 = None
    mul_41: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_23);  mul_40 = None
    add_42: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_24);  mul_41 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_4: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_105: "i64[1, 14]" = torch.ops.aten.view.default(iota_8, [1, -1]);  iota_8 = None
    iota_9: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_106: "i64[14, 1]" = torch.ops.aten.view.default(iota_9, [-1, 1]);  iota_9 = None
    sub_25: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_105, view_106);  view_105 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_4: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_25, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_24: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_25, 1);  sub_25 = None
    expand_29: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_24, [14, 14, 14]);  unsqueeze_24 = None
    clone_62: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_107: "i64[196, 14]" = torch.ops.aten.view.default(clone_62, [196, 14]);  clone_62 = None
    unsqueeze_25: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_107, 2);  view_107 = None
    expand_30: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_25, [196, 14, 14]);  unsqueeze_25 = None
    clone_63: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_108: "i64[196, 196]" = torch.ops.aten.view.default(clone_63, [196, 196]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_9: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_4, 2)
    pow_10: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_108, 2)
    add_43: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_9, pow_10);  pow_9 = pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_26: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_43, 0);  add_43 = None
    slice_133: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_4, 0, 0, 9223372036854775807)
    slice_134: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_133, 1, 0, 9223372036854775807);  slice_133 = None
    slice_135: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_134, 2, 0, 9223372036854775807);  slice_134 = None
    select_40: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_135, 3, 2);  slice_135 = None
    copy_12: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_40, unsqueeze_26);  select_40 = unsqueeze_26 = None
    slice_136: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_4, 0, 0, 9223372036854775807)
    slice_137: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_136, 1, 0, 9223372036854775807)
    slice_138: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_137, 2, 0, 9223372036854775807)
    select_scatter_12: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_138, copy_12, 3, 2);  slice_138 = copy_12 = None
    slice_scatter_36: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_137, select_scatter_12, 2, 0, 9223372036854775807);  slice_137 = select_scatter_12 = None
    slice_scatter_37: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_136, slice_scatter_36, 1, 0, 9223372036854775807);  slice_136 = slice_scatter_36 = None
    slice_scatter_38: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_37, 0, 0, 9223372036854775807);  full_4 = slice_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_27: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
    slice_145: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 0, 9223372036854775807)
    slice_146: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_145, 1, 0, 9223372036854775807);  slice_145 = None
    slice_147: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_146, 2, 0, 9223372036854775807);  slice_146 = None
    select_43: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_147, 3, 1);  slice_147 = None
    copy_13: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_43, unsqueeze_27);  select_43 = unsqueeze_27 = None
    slice_148: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 0, 9223372036854775807)
    slice_149: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_148, 1, 0, 9223372036854775807)
    slice_150: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_149, 2, 0, 9223372036854775807)
    select_scatter_13: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_150, copy_13, 3, 1);  slice_150 = copy_13 = None
    slice_scatter_39: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_149, select_scatter_13, 2, 0, 9223372036854775807);  slice_149 = select_scatter_13 = None
    slice_scatter_40: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_148, slice_scatter_39, 1, 0, 9223372036854775807);  slice_148 = slice_scatter_39 = None
    slice_scatter_41: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_38, slice_scatter_40, 0, 0, 9223372036854775807);  slice_scatter_38 = slice_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_28: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_4, 0);  repeat_4 = None
    slice_157: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 0, 0, 9223372036854775807)
    slice_158: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_157, 1, 0, 9223372036854775807);  slice_157 = None
    slice_159: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_158, 2, 0, 9223372036854775807);  slice_158 = None
    select_46: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_159, 3, 0);  slice_159 = None
    copy_14: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_46, unsqueeze_28);  select_46 = unsqueeze_28 = None
    slice_160: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 0, 0, 9223372036854775807)
    slice_161: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_160, 1, 0, 9223372036854775807)
    slice_162: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_161, 2, 0, 9223372036854775807)
    select_scatter_14: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_162, copy_14, 3, 0);  slice_162 = copy_14 = None
    slice_scatter_42: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_161, select_scatter_14, 2, 0, 9223372036854775807);  slice_161 = select_scatter_14 = None
    slice_scatter_43: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_160, slice_scatter_42, 1, 0, 9223372036854775807);  slice_160 = slice_scatter_42 = None
    slice_scatter_44: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_41, slice_scatter_43, 0, 0, 9223372036854775807);  slice_scatter_41 = slice_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_4: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_44, device(type='cuda', index=0));  slice_scatter_44 = None
    convert_element_type_4: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_4, torch.float32);  device_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_45: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_109: "f32[1568, 768]" = torch.ops.aten.view.default(add_42, [1568, 768])
    mm_12: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_109, permute_45)
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_12, [8, 196, 1536]);  mm_12 = None
    view_111: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_110, [8, 196, 2, 16, 48]);  view_110 = None
    permute_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_48: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 0)
    select_49: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_46, 0, 1);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_31: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_4, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_47: "f32[3, 16]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    clone_64: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_112: "f32[307328, 3]" = torch.ops.aten.view.default(clone_64, [307328, 3]);  clone_64 = None
    mm_13: "f32[307328, 16]" = torch.ops.aten.mm.default(view_112, permute_47);  permute_47 = None
    view_113: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_13, [8, 196, 196, 16]);  mm_13 = None
    add_44: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_113, primals_107);  view_113 = primals_107 = None
    permute_48: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_49: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_32: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_48, [8, 16, 196, 48]);  select_48 = None
    clone_65: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_114: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_65, [128, 196, 48]);  clone_65 = None
    expand_33: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_49, [8, 16, 48, 196]);  permute_49 = None
    clone_66: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_115: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_66, [128, 48, 196]);  clone_66 = None
    bmm_8: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 16, 196, 196]);  bmm_8 = None
    mul_42: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_116, 0.14433756729740643);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_8: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_26: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_42, amax_8);  mul_42 = amax_8 = None
    exp_8: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_13);  exp_8 = sum_13 = None
    alias_16: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_67: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax_9: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_67, [-1], True)
    sub_27: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_67, amax_9);  clone_67 = amax_9 = None
    exp_9: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_14);  exp_9 = sum_14 = None
    alias_17: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_117: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_25, [1, -1, 1, 1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117)
    alias_18: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_8)
    sub_28: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_8);  sigmoid_8 = None
    mul_43: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_28, div_12)
    sigmoid_9: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117);  view_117 = None
    alias_19: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    mul_44: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_9, div_13)
    add_45: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_15: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_45, [-1])
    unsqueeze_29: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_15, -1);  sum_15 = None
    clone_68: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_45)
    div_14: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_45, unsqueeze_29);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_69: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    view_118: "f32[1568, 768]" = torch.ops.aten.view.default(add_42, [1568, 768]);  add_42 = None
    mm_14: "f32[1568, 768]" = torch.ops.aten.mm.default(view_118, permute_50)
    view_119: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_14, [8, 196, 768]);  mm_14 = None
    view_120: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_119, [8, 196, 16, 48]);  view_119 = None
    permute_51: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_69, [8, 16, 196, 196]);  clone_69 = None
    view_121: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_34, [128, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_51, [8, 16, 196, 48]);  permute_51 = None
    clone_70: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_122: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_70, [128, 196, 48]);  clone_70 = None
    bmm_9: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_9, [8, 16, 196, 48]);  bmm_9 = None
    permute_52: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_71: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_124: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_71, [8, 196, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.view.default(view_124, [1568, 768]);  view_124 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_12: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_110, view_125, permute_53);  primals_110 = None
    view_126: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_12, [8, 196, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_72: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_40, clone_72);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_73: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_73, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_73, getitem_19);  clone_73 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_9);  sub_29 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_26);  mul_45 = None
    add_48: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_27);  mul_46 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.view.default(add_48, [1568, 768]);  add_48 = None
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_13: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_112, view_127, permute_54);  primals_112 = None
    view_128: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 196, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_4: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_49: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_49: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_49);  mul_47 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_74: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_74, [1568, 3072]);  clone_74 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_14: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_114, view_129, permute_55);  primals_114 = None
    view_130: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_14, [8, 196, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_75: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_130);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_46, clone_75);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_76: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_76, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_51: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_30: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_76, getitem_21);  clone_76 = None
    mul_50: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_10);  sub_30 = None
    mul_51: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_28);  mul_50 = None
    add_52: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_29);  mul_51 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_5: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_131: "i64[1, 14]" = torch.ops.aten.view.default(iota_10, [1, -1]);  iota_10 = None
    iota_11: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_132: "i64[14, 1]" = torch.ops.aten.view.default(iota_11, [-1, 1]);  iota_11 = None
    sub_31: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_131, view_132);  view_131 = view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_5: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_31, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_30: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_31, 1);  sub_31 = None
    expand_36: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_30, [14, 14, 14]);  unsqueeze_30 = None
    clone_77: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_133: "i64[196, 14]" = torch.ops.aten.view.default(clone_77, [196, 14]);  clone_77 = None
    unsqueeze_31: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_133, 2);  view_133 = None
    expand_37: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_31, [196, 14, 14]);  unsqueeze_31 = None
    clone_78: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_134: "i64[196, 196]" = torch.ops.aten.view.default(clone_78, [196, 196]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_11: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_5, 2)
    pow_12: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_134, 2)
    add_53: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_11, pow_12);  pow_11 = pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_32: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_53, 0);  add_53 = None
    slice_166: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_167: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_166, 1, 0, 9223372036854775807);  slice_166 = None
    slice_168: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_167, 2, 0, 9223372036854775807);  slice_167 = None
    select_50: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_168, 3, 2);  slice_168 = None
    copy_15: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_50, unsqueeze_32);  select_50 = unsqueeze_32 = None
    slice_169: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_170: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_169, 1, 0, 9223372036854775807)
    slice_171: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_170, 2, 0, 9223372036854775807)
    select_scatter_15: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_171, copy_15, 3, 2);  slice_171 = copy_15 = None
    slice_scatter_45: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_170, select_scatter_15, 2, 0, 9223372036854775807);  slice_170 = select_scatter_15 = None
    slice_scatter_46: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_169, slice_scatter_45, 1, 0, 9223372036854775807);  slice_169 = slice_scatter_45 = None
    slice_scatter_47: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_46, 0, 0, 9223372036854775807);  full_5 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_33: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_134, 0);  view_134 = None
    slice_178: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    slice_179: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_178, 1, 0, 9223372036854775807);  slice_178 = None
    slice_180: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_179, 2, 0, 9223372036854775807);  slice_179 = None
    select_53: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_180, 3, 1);  slice_180 = None
    copy_16: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_53, unsqueeze_33);  select_53 = unsqueeze_33 = None
    slice_181: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    slice_182: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_181, 1, 0, 9223372036854775807)
    slice_183: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_182, 2, 0, 9223372036854775807)
    select_scatter_16: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_183, copy_16, 3, 1);  slice_183 = copy_16 = None
    slice_scatter_48: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_182, select_scatter_16, 2, 0, 9223372036854775807);  slice_182 = select_scatter_16 = None
    slice_scatter_49: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_181, slice_scatter_48, 1, 0, 9223372036854775807);  slice_181 = slice_scatter_48 = None
    slice_scatter_50: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_47, slice_scatter_49, 0, 0, 9223372036854775807);  slice_scatter_47 = slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_34: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_5, 0);  repeat_5 = None
    slice_190: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_191: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_190, 1, 0, 9223372036854775807);  slice_190 = None
    slice_192: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_191, 2, 0, 9223372036854775807);  slice_191 = None
    select_56: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_192, 3, 0);  slice_192 = None
    copy_17: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_56, unsqueeze_34);  select_56 = unsqueeze_34 = None
    slice_193: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_194: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_193, 1, 0, 9223372036854775807)
    slice_195: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_194, 2, 0, 9223372036854775807)
    select_scatter_17: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_195, copy_17, 3, 0);  slice_195 = copy_17 = None
    slice_scatter_51: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_194, select_scatter_17, 2, 0, 9223372036854775807);  slice_194 = select_scatter_17 = None
    slice_scatter_52: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_193, slice_scatter_51, 1, 0, 9223372036854775807);  slice_193 = slice_scatter_51 = None
    slice_scatter_53: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_52, 0, 0, 9223372036854775807);  slice_scatter_50 = slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_5: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_53, device(type='cuda', index=0));  slice_scatter_53 = None
    convert_element_type_5: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_5, torch.float32);  device_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_56: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    view_135: "f32[1568, 768]" = torch.ops.aten.view.default(add_52, [1568, 768])
    mm_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_135, permute_56)
    view_136: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_15, [8, 196, 1536]);  mm_15 = None
    view_137: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_136, [8, 196, 2, 16, 48]);  view_136 = None
    permute_57: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_137, [2, 0, 3, 1, 4]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_58: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 0)
    select_59: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_57, 0, 1);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_38: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_5, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_58: "f32[3, 16]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    clone_79: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_138: "f32[307328, 3]" = torch.ops.aten.view.default(clone_79, [307328, 3]);  clone_79 = None
    mm_16: "f32[307328, 16]" = torch.ops.aten.mm.default(view_138, permute_58);  permute_58 = None
    view_139: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_16, [8, 196, 196, 16]);  mm_16 = None
    add_54: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_139, primals_117);  view_139 = primals_117 = None
    permute_59: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_60: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_59, [0, 1, 3, 2]);  select_59 = None
    expand_39: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_58, [8, 16, 196, 48]);  select_58 = None
    clone_80: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_140: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_80, [128, 196, 48]);  clone_80 = None
    expand_40: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_60, [8, 16, 48, 196]);  permute_60 = None
    clone_81: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_141: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_81, [128, 48, 196]);  clone_81 = None
    bmm_10: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_140, view_141)
    view_142: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 16, 196, 196]);  bmm_10 = None
    mul_52: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_142, 0.14433756729740643);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_10: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_32: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_52, amax_10);  mul_52 = amax_10 = None
    exp_10: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_16: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_15: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_16);  exp_10 = sum_16 = None
    alias_20: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_82: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    amax_11: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_82, [-1], True)
    sub_33: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_82, amax_11);  clone_82 = amax_11 = None
    exp_11: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_16: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_17);  exp_11 = sum_17 = None
    alias_21: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_143: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_30, [1, -1, 1, 1]);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143)
    alias_22: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_10)
    sub_34: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_10);  sigmoid_10 = None
    mul_53: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_34, div_15)
    sigmoid_11: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143);  view_143 = None
    alias_23: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    mul_54: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_11, div_16)
    add_55: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_18: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_55, [-1])
    unsqueeze_35: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_18, -1);  sum_18 = None
    clone_83: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_55)
    div_17: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_55, unsqueeze_35);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_84: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    view_144: "f32[1568, 768]" = torch.ops.aten.view.default(add_52, [1568, 768]);  add_52 = None
    mm_17: "f32[1568, 768]" = torch.ops.aten.mm.default(view_144, permute_61)
    view_145: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_17, [8, 196, 768]);  mm_17 = None
    view_146: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_145, [8, 196, 16, 48]);  view_145 = None
    permute_62: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_41: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_84, [8, 16, 196, 196]);  clone_84 = None
    view_147: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_41, [128, 196, 196]);  expand_41 = None
    expand_42: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_62, [8, 16, 196, 48]);  permute_62 = None
    clone_85: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_148: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_85, [128, 196, 48]);  clone_85 = None
    bmm_11: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_147, view_148)
    view_149: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_11, [8, 16, 196, 48]);  bmm_11 = None
    permute_63: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_86: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_150: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_86, [8, 196, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.view.default(view_150, [1568, 768]);  view_150 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_15: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_120, view_151, permute_64);  primals_120 = None
    view_152: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_15, [8, 196, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_87: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_56: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_50, clone_87);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_88: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_88, getitem_23);  clone_88 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_11);  sub_35 = None
    mul_56: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_31);  mul_55 = None
    add_58: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_32);  mul_56 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_153: "f32[1568, 768]" = torch.ops.aten.view.default(add_58, [1568, 768]);  add_58 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_16: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_122, view_153, permute_65);  primals_122 = None
    view_154: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 196, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_57: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476)
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_59: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_59: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_59);  mul_57 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_89: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_155: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_89, [1568, 3072]);  clone_89 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_17: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_124, view_155, permute_66);  primals_124 = None
    view_156: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_17, [8, 196, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_90: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_56, clone_90);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_91: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_36: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_91, getitem_25);  clone_91 = None
    mul_60: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_12);  sub_36 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_33);  mul_60 = None
    add_62: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_34);  mul_61 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_6: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_157: "i64[1, 14]" = torch.ops.aten.view.default(iota_12, [1, -1]);  iota_12 = None
    iota_13: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_158: "i64[14, 1]" = torch.ops.aten.view.default(iota_13, [-1, 1]);  iota_13 = None
    sub_37: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_157, view_158);  view_157 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_6: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_37, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_36: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_37, 1);  sub_37 = None
    expand_43: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_36, [14, 14, 14]);  unsqueeze_36 = None
    clone_92: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_159: "i64[196, 14]" = torch.ops.aten.view.default(clone_92, [196, 14]);  clone_92 = None
    unsqueeze_37: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_159, 2);  view_159 = None
    expand_44: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_37, [196, 14, 14]);  unsqueeze_37 = None
    clone_93: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_160: "i64[196, 196]" = torch.ops.aten.view.default(clone_93, [196, 196]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_13: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_6, 2)
    pow_14: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_160, 2)
    add_63: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_13, pow_14);  pow_13 = pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_38: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_63, 0);  add_63 = None
    slice_199: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_6, 0, 0, 9223372036854775807)
    slice_200: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_199, 1, 0, 9223372036854775807);  slice_199 = None
    slice_201: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_200, 2, 0, 9223372036854775807);  slice_200 = None
    select_60: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_201, 3, 2);  slice_201 = None
    copy_18: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_60, unsqueeze_38);  select_60 = unsqueeze_38 = None
    slice_202: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_6, 0, 0, 9223372036854775807)
    slice_203: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_202, 1, 0, 9223372036854775807)
    slice_204: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_203, 2, 0, 9223372036854775807)
    select_scatter_18: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_204, copy_18, 3, 2);  slice_204 = copy_18 = None
    slice_scatter_54: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_203, select_scatter_18, 2, 0, 9223372036854775807);  slice_203 = select_scatter_18 = None
    slice_scatter_55: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_202, slice_scatter_54, 1, 0, 9223372036854775807);  slice_202 = slice_scatter_54 = None
    slice_scatter_56: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_55, 0, 0, 9223372036854775807);  full_6 = slice_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_39: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_160, 0);  view_160 = None
    slice_211: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 0, 9223372036854775807)
    slice_212: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_211, 1, 0, 9223372036854775807);  slice_211 = None
    slice_213: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_212, 2, 0, 9223372036854775807);  slice_212 = None
    select_63: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_213, 3, 1);  slice_213 = None
    copy_19: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_63, unsqueeze_39);  select_63 = unsqueeze_39 = None
    slice_214: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 0, 9223372036854775807)
    slice_215: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_214, 1, 0, 9223372036854775807)
    slice_216: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_215, 2, 0, 9223372036854775807)
    select_scatter_19: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_216, copy_19, 3, 1);  slice_216 = copy_19 = None
    slice_scatter_57: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_215, select_scatter_19, 2, 0, 9223372036854775807);  slice_215 = select_scatter_19 = None
    slice_scatter_58: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_214, slice_scatter_57, 1, 0, 9223372036854775807);  slice_214 = slice_scatter_57 = None
    slice_scatter_59: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_56, slice_scatter_58, 0, 0, 9223372036854775807);  slice_scatter_56 = slice_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_40: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_6, 0);  repeat_6 = None
    slice_223: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 0, 9223372036854775807)
    slice_224: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_223, 1, 0, 9223372036854775807);  slice_223 = None
    slice_225: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_224, 2, 0, 9223372036854775807);  slice_224 = None
    select_66: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_225, 3, 0);  slice_225 = None
    copy_20: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_66, unsqueeze_40);  select_66 = unsqueeze_40 = None
    slice_226: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 0, 9223372036854775807)
    slice_227: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_226, 1, 0, 9223372036854775807)
    slice_228: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_227, 2, 0, 9223372036854775807)
    select_scatter_20: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_228, copy_20, 3, 0);  slice_228 = copy_20 = None
    slice_scatter_60: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_227, select_scatter_20, 2, 0, 9223372036854775807);  slice_227 = select_scatter_20 = None
    slice_scatter_61: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_226, slice_scatter_60, 1, 0, 9223372036854775807);  slice_226 = slice_scatter_60 = None
    slice_scatter_62: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_59, slice_scatter_61, 0, 0, 9223372036854775807);  slice_scatter_59 = slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_6: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_62, device(type='cuda', index=0));  slice_scatter_62 = None
    convert_element_type_6: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_6, torch.float32);  device_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_67: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_161: "f32[1568, 768]" = torch.ops.aten.view.default(add_62, [1568, 768])
    mm_18: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_161, permute_67)
    view_162: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_18, [8, 196, 1536]);  mm_18 = None
    view_163: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_162, [8, 196, 2, 16, 48]);  view_162 = None
    permute_68: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_68: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 0)
    select_69: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_68, 0, 1);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_45: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_6, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_69: "f32[3, 16]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    clone_94: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_164: "f32[307328, 3]" = torch.ops.aten.view.default(clone_94, [307328, 3]);  clone_94 = None
    mm_19: "f32[307328, 16]" = torch.ops.aten.mm.default(view_164, permute_69);  permute_69 = None
    view_165: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_19, [8, 196, 196, 16]);  mm_19 = None
    add_64: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_165, primals_127);  view_165 = primals_127 = None
    permute_70: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_64, [0, 3, 1, 2]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_71: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_69, [0, 1, 3, 2]);  select_69 = None
    expand_46: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 196, 48]);  select_68 = None
    clone_95: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_166: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_95, [128, 196, 48]);  clone_95 = None
    expand_47: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_71, [8, 16, 48, 196]);  permute_71 = None
    clone_96: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_167: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_96, [128, 48, 196]);  clone_96 = None
    bmm_12: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 16, 196, 196]);  bmm_12 = None
    mul_62: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_168, 0.14433756729740643);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_12: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_62, [-1], True)
    sub_38: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_62, amax_12);  mul_62 = amax_12 = None
    exp_12: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_18: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_19);  exp_12 = sum_19 = None
    alias_24: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_97: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    amax_13: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_97, [-1], True)
    sub_39: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_97, amax_13);  clone_97 = amax_13 = None
    exp_13: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_19: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_20);  exp_13 = sum_20 = None
    alias_25: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_169: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_35, [1, -1, 1, 1]);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169)
    alias_26: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_12)
    sub_40: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_12);  sigmoid_12 = None
    mul_63: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_40, div_18)
    sigmoid_13: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
    alias_27: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_13)
    mul_64: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_13, div_19)
    add_65: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_21: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_65, [-1])
    unsqueeze_41: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_21, -1);  sum_21 = None
    clone_98: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_65)
    div_20: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_65, unsqueeze_41);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_99: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    view_170: "f32[1568, 768]" = torch.ops.aten.view.default(add_62, [1568, 768]);  add_62 = None
    mm_20: "f32[1568, 768]" = torch.ops.aten.mm.default(view_170, permute_72)
    view_171: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_20, [8, 196, 768]);  mm_20 = None
    view_172: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_171, [8, 196, 16, 48]);  view_171 = None
    permute_73: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_48: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_99, [8, 16, 196, 196]);  clone_99 = None
    view_173: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_48, [128, 196, 196]);  expand_48 = None
    expand_49: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_73, [8, 16, 196, 48]);  permute_73 = None
    clone_100: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_174: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_100, [128, 196, 48]);  clone_100 = None
    bmm_13: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_173, view_174)
    view_175: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_13, [8, 16, 196, 48]);  bmm_13 = None
    permute_74: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    clone_101: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_176: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_101, [8, 196, 768]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_177: "f32[1568, 768]" = torch.ops.aten.view.default(view_176, [1568, 768]);  view_176 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_18: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_130, view_177, permute_75);  primals_130 = None
    view_178: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_18, [8, 196, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_102: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_60, clone_102);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_103: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_103, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_103, getitem_27);  clone_103 = None
    mul_65: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_13);  sub_41 = None
    mul_66: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_36);  mul_65 = None
    add_68: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_37);  mul_66 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.view.default(add_68, [1568, 768]);  add_68 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_19: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_132, view_179, permute_76);  primals_132 = None
    view_180: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 196, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.5)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476)
    erf_6: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_69: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_104: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_181: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_104, [1568, 3072]);  clone_104 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_20: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_134, view_181, permute_77);  primals_134 = None
    view_182: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_20, [8, 196, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_105: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_182);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_66, clone_105);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_106: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_106, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_71: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_42: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_106, getitem_29);  clone_106 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_14);  sub_42 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_38);  mul_70 = None
    add_72: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_39);  mul_71 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_7: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_183: "i64[1, 14]" = torch.ops.aten.view.default(iota_14, [1, -1]);  iota_14 = None
    iota_15: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_184: "i64[14, 1]" = torch.ops.aten.view.default(iota_15, [-1, 1]);  iota_15 = None
    sub_43: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_183, view_184);  view_183 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_7: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_43, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_42: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_43, 1);  sub_43 = None
    expand_50: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_42, [14, 14, 14]);  unsqueeze_42 = None
    clone_107: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_185: "i64[196, 14]" = torch.ops.aten.view.default(clone_107, [196, 14]);  clone_107 = None
    unsqueeze_43: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_185, 2);  view_185 = None
    expand_51: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_43, [196, 14, 14]);  unsqueeze_43 = None
    clone_108: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_186: "i64[196, 196]" = torch.ops.aten.view.default(clone_108, [196, 196]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_15: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_7, 2)
    pow_16: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_186, 2)
    add_73: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_15, pow_16);  pow_15 = pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_44: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_73, 0);  add_73 = None
    slice_232: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_7, 0, 0, 9223372036854775807)
    slice_233: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_232, 1, 0, 9223372036854775807);  slice_232 = None
    slice_234: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_233, 2, 0, 9223372036854775807);  slice_233 = None
    select_70: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_234, 3, 2);  slice_234 = None
    copy_21: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_70, unsqueeze_44);  select_70 = unsqueeze_44 = None
    slice_235: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_7, 0, 0, 9223372036854775807)
    slice_236: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_235, 1, 0, 9223372036854775807)
    slice_237: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_236, 2, 0, 9223372036854775807)
    select_scatter_21: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_237, copy_21, 3, 2);  slice_237 = copy_21 = None
    slice_scatter_63: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_236, select_scatter_21, 2, 0, 9223372036854775807);  slice_236 = select_scatter_21 = None
    slice_scatter_64: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_235, slice_scatter_63, 1, 0, 9223372036854775807);  slice_235 = slice_scatter_63 = None
    slice_scatter_65: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_64, 0, 0, 9223372036854775807);  full_7 = slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_45: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_186, 0);  view_186 = None
    slice_244: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 0, 9223372036854775807)
    slice_245: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_244, 1, 0, 9223372036854775807);  slice_244 = None
    slice_246: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_245, 2, 0, 9223372036854775807);  slice_245 = None
    select_73: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_246, 3, 1);  slice_246 = None
    copy_22: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_73, unsqueeze_45);  select_73 = unsqueeze_45 = None
    slice_247: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 0, 9223372036854775807)
    slice_248: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_247, 1, 0, 9223372036854775807)
    slice_249: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 9223372036854775807)
    select_scatter_22: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_249, copy_22, 3, 1);  slice_249 = copy_22 = None
    slice_scatter_66: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_248, select_scatter_22, 2, 0, 9223372036854775807);  slice_248 = select_scatter_22 = None
    slice_scatter_67: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_247, slice_scatter_66, 1, 0, 9223372036854775807);  slice_247 = slice_scatter_66 = None
    slice_scatter_68: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_65, slice_scatter_67, 0, 0, 9223372036854775807);  slice_scatter_65 = slice_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_46: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_7, 0);  repeat_7 = None
    slice_256: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 0, 9223372036854775807)
    slice_257: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_256, 1, 0, 9223372036854775807);  slice_256 = None
    slice_258: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_257, 2, 0, 9223372036854775807);  slice_257 = None
    select_76: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_258, 3, 0);  slice_258 = None
    copy_23: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_76, unsqueeze_46);  select_76 = unsqueeze_46 = None
    slice_259: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 0, 9223372036854775807)
    slice_260: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_259, 1, 0, 9223372036854775807)
    slice_261: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_260, 2, 0, 9223372036854775807)
    select_scatter_23: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_261, copy_23, 3, 0);  slice_261 = copy_23 = None
    slice_scatter_69: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_260, select_scatter_23, 2, 0, 9223372036854775807);  slice_260 = select_scatter_23 = None
    slice_scatter_70: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_259, slice_scatter_69, 1, 0, 9223372036854775807);  slice_259 = slice_scatter_69 = None
    slice_scatter_71: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_68, slice_scatter_70, 0, 0, 9223372036854775807);  slice_scatter_68 = slice_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_7: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_71, device(type='cuda', index=0));  slice_scatter_71 = None
    convert_element_type_7: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_7, torch.float32);  device_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_78: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_187: "f32[1568, 768]" = torch.ops.aten.view.default(add_72, [1568, 768])
    mm_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_187, permute_78)
    view_188: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_21, [8, 196, 1536]);  mm_21 = None
    view_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_188, [8, 196, 2, 16, 48]);  view_188 = None
    permute_79: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_189, [2, 0, 3, 1, 4]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_78: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 0)
    select_79: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_79, 0, 1);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_52: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_7, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_80: "f32[3, 16]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    clone_109: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_190: "f32[307328, 3]" = torch.ops.aten.view.default(clone_109, [307328, 3]);  clone_109 = None
    mm_22: "f32[307328, 16]" = torch.ops.aten.mm.default(view_190, permute_80);  permute_80 = None
    view_191: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_22, [8, 196, 196, 16]);  mm_22 = None
    add_74: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_191, primals_137);  view_191 = primals_137 = None
    permute_81: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_74, [0, 3, 1, 2]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_53: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_78, [8, 16, 196, 48]);  select_78 = None
    clone_110: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_192: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_110, [128, 196, 48]);  clone_110 = None
    expand_54: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_82, [8, 16, 48, 196]);  permute_82 = None
    clone_111: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_193: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_111, [128, 48, 196]);  clone_111 = None
    bmm_14: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_192, view_193)
    view_194: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 16, 196, 196]);  bmm_14 = None
    mul_72: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_194, 0.14433756729740643);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_14: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_72, [-1], True)
    sub_44: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_72, amax_14);  mul_72 = amax_14 = None
    exp_14: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_22: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_21: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_22);  exp_14 = sum_22 = None
    alias_28: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_112: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    amax_15: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_112, [-1], True)
    sub_45: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_112, amax_15);  clone_112 = amax_15 = None
    exp_15: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_22: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_23);  exp_15 = sum_23 = None
    alias_29: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_195: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_40, [1, -1, 1, 1]);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195)
    alias_30: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_14)
    sub_46: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_14);  sigmoid_14 = None
    mul_73: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_46, div_21)
    sigmoid_15: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195);  view_195 = None
    alias_31: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    mul_74: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_15, div_22)
    add_75: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_24: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_75, [-1])
    unsqueeze_47: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_24, -1);  sum_24 = None
    clone_113: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_75)
    div_23: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_75, unsqueeze_47);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_114: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    view_196: "f32[1568, 768]" = torch.ops.aten.view.default(add_72, [1568, 768]);  add_72 = None
    mm_23: "f32[1568, 768]" = torch.ops.aten.mm.default(view_196, permute_83)
    view_197: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_23, [8, 196, 768]);  mm_23 = None
    view_198: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_197, [8, 196, 16, 48]);  view_197 = None
    permute_84: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_55: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_114, [8, 16, 196, 196]);  clone_114 = None
    view_199: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_55, [128, 196, 196]);  expand_55 = None
    expand_56: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_84, [8, 16, 196, 48]);  permute_84 = None
    clone_115: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_200: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_115, [128, 196, 48]);  clone_115 = None
    bmm_15: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_199, view_200)
    view_201: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_15, [8, 16, 196, 48]);  bmm_15 = None
    permute_85: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    clone_116: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_116, [8, 196, 768]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_203: "f32[1568, 768]" = torch.ops.aten.view.default(view_202, [1568, 768]);  view_202 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_21: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_140, view_203, permute_86);  primals_140 = None
    view_204: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_21, [8, 196, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_204);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_76: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_70, clone_117);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_118: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_118, getitem_31);  clone_118 = None
    mul_75: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_15);  sub_47 = None
    mul_76: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_41);  mul_75 = None
    add_78: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_42);  mul_76 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_205: "f32[1568, 768]" = torch.ops.aten.view.default(add_78, [1568, 768]);  add_78 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_22: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_142, view_205, permute_87);  primals_142 = None
    view_206: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 196, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_79: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_79);  mul_77 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_119: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_207: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_119, [1568, 3072]);  clone_119 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_23: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_144, view_207, permute_88);  primals_144 = None
    view_208: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_23, [8, 196, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_120: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_80: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_76, clone_120);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_121: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_121, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_48: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_121, getitem_33);  clone_121 = None
    mul_80: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_16);  sub_48 = None
    mul_81: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_43);  mul_80 = None
    add_82: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_44);  mul_81 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_8: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_209: "i64[1, 14]" = torch.ops.aten.view.default(iota_16, [1, -1]);  iota_16 = None
    iota_17: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_210: "i64[14, 1]" = torch.ops.aten.view.default(iota_17, [-1, 1]);  iota_17 = None
    sub_49: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_209, view_210);  view_209 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_8: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_49, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_48: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_49, 1);  sub_49 = None
    expand_57: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_48, [14, 14, 14]);  unsqueeze_48 = None
    clone_122: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_211: "i64[196, 14]" = torch.ops.aten.view.default(clone_122, [196, 14]);  clone_122 = None
    unsqueeze_49: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_211, 2);  view_211 = None
    expand_58: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_49, [196, 14, 14]);  unsqueeze_49 = None
    clone_123: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_212: "i64[196, 196]" = torch.ops.aten.view.default(clone_123, [196, 196]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_17: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_8, 2)
    pow_18: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_212, 2)
    add_83: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_17, pow_18);  pow_17 = pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_50: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_83, 0);  add_83 = None
    slice_265: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_8, 0, 0, 9223372036854775807)
    slice_266: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_265, 1, 0, 9223372036854775807);  slice_265 = None
    slice_267: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_266, 2, 0, 9223372036854775807);  slice_266 = None
    select_80: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_267, 3, 2);  slice_267 = None
    copy_24: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_80, unsqueeze_50);  select_80 = unsqueeze_50 = None
    slice_268: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_8, 0, 0, 9223372036854775807)
    slice_269: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_268, 1, 0, 9223372036854775807)
    slice_270: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_269, 2, 0, 9223372036854775807)
    select_scatter_24: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_270, copy_24, 3, 2);  slice_270 = copy_24 = None
    slice_scatter_72: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_269, select_scatter_24, 2, 0, 9223372036854775807);  slice_269 = select_scatter_24 = None
    slice_scatter_73: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_268, slice_scatter_72, 1, 0, 9223372036854775807);  slice_268 = slice_scatter_72 = None
    slice_scatter_74: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_73, 0, 0, 9223372036854775807);  full_8 = slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_51: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_212, 0);  view_212 = None
    slice_277: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 0, 0, 9223372036854775807)
    slice_278: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_277, 1, 0, 9223372036854775807);  slice_277 = None
    slice_279: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_278, 2, 0, 9223372036854775807);  slice_278 = None
    select_83: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_279, 3, 1);  slice_279 = None
    copy_25: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_83, unsqueeze_51);  select_83 = unsqueeze_51 = None
    slice_280: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 0, 0, 9223372036854775807)
    slice_281: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_280, 1, 0, 9223372036854775807)
    slice_282: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_281, 2, 0, 9223372036854775807)
    select_scatter_25: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_282, copy_25, 3, 1);  slice_282 = copy_25 = None
    slice_scatter_75: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_281, select_scatter_25, 2, 0, 9223372036854775807);  slice_281 = select_scatter_25 = None
    slice_scatter_76: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_280, slice_scatter_75, 1, 0, 9223372036854775807);  slice_280 = slice_scatter_75 = None
    slice_scatter_77: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_74, slice_scatter_76, 0, 0, 9223372036854775807);  slice_scatter_74 = slice_scatter_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_52: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_8, 0);  repeat_8 = None
    slice_289: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 0, 0, 9223372036854775807)
    slice_290: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_289, 1, 0, 9223372036854775807);  slice_289 = None
    slice_291: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_290, 2, 0, 9223372036854775807);  slice_290 = None
    select_86: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_291, 3, 0);  slice_291 = None
    copy_26: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_86, unsqueeze_52);  select_86 = unsqueeze_52 = None
    slice_292: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 0, 0, 9223372036854775807)
    slice_293: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_292, 1, 0, 9223372036854775807)
    slice_294: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_293, 2, 0, 9223372036854775807)
    select_scatter_26: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_294, copy_26, 3, 0);  slice_294 = copy_26 = None
    slice_scatter_78: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_293, select_scatter_26, 2, 0, 9223372036854775807);  slice_293 = select_scatter_26 = None
    slice_scatter_79: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_292, slice_scatter_78, 1, 0, 9223372036854775807);  slice_292 = slice_scatter_78 = None
    slice_scatter_80: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_77, slice_scatter_79, 0, 0, 9223372036854775807);  slice_scatter_77 = slice_scatter_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_8: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_80, device(type='cuda', index=0));  slice_scatter_80 = None
    convert_element_type_8: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_8, torch.float32);  device_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_89: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    view_213: "f32[1568, 768]" = torch.ops.aten.view.default(add_82, [1568, 768])
    mm_24: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_213, permute_89)
    view_214: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_24, [8, 196, 1536]);  mm_24 = None
    view_215: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_214, [8, 196, 2, 16, 48]);  view_214 = None
    permute_90: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_88: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 0)
    select_89: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_90, 0, 1);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_59: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_8, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_91: "f32[3, 16]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    clone_124: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_216: "f32[307328, 3]" = torch.ops.aten.view.default(clone_124, [307328, 3]);  clone_124 = None
    mm_25: "f32[307328, 16]" = torch.ops.aten.mm.default(view_216, permute_91);  permute_91 = None
    view_217: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_25, [8, 196, 196, 16]);  mm_25 = None
    add_84: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_217, primals_147);  view_217 = primals_147 = None
    permute_92: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_93: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_89, [0, 1, 3, 2]);  select_89 = None
    expand_60: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_88, [8, 16, 196, 48]);  select_88 = None
    clone_125: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_218: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_125, [128, 196, 48]);  clone_125 = None
    expand_61: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_93, [8, 16, 48, 196]);  permute_93 = None
    clone_126: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_219: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_126, [128, 48, 196]);  clone_126 = None
    bmm_16: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_218, view_219)
    view_220: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 16, 196, 196]);  bmm_16 = None
    mul_82: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_220, 0.14433756729740643);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_16: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_82, [-1], True)
    sub_50: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_82, amax_16);  mul_82 = amax_16 = None
    exp_16: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_25: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_24: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_25);  exp_16 = sum_25 = None
    alias_32: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_127: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    amax_17: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_127, [-1], True)
    sub_51: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_127, amax_17);  clone_127 = amax_17 = None
    exp_17: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_25: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_26);  exp_17 = sum_26 = None
    alias_33: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_221: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_45, [1, -1, 1, 1]);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221)
    alias_34: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_16)
    sub_52: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_16);  sigmoid_16 = None
    mul_83: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_52, div_24)
    sigmoid_17: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221);  view_221 = None
    alias_35: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_17)
    mul_84: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_17, div_25)
    add_85: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_27: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_85, [-1])
    unsqueeze_53: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_27, -1);  sum_27 = None
    clone_128: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_85)
    div_26: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_85, unsqueeze_53);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_129: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    view_222: "f32[1568, 768]" = torch.ops.aten.view.default(add_82, [1568, 768]);  add_82 = None
    mm_26: "f32[1568, 768]" = torch.ops.aten.mm.default(view_222, permute_94)
    view_223: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_26, [8, 196, 768]);  mm_26 = None
    view_224: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_223, [8, 196, 16, 48]);  view_223 = None
    permute_95: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_62: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_129, [8, 16, 196, 196]);  clone_129 = None
    view_225: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_62, [128, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 196, 48]);  permute_95 = None
    clone_130: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_226: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_130, [128, 196, 48]);  clone_130 = None
    bmm_17: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_225, view_226)
    view_227: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_17, [8, 16, 196, 48]);  bmm_17 = None
    permute_96: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_131: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_228: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_131, [8, 196, 768]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_229: "f32[1568, 768]" = torch.ops.aten.view.default(view_228, [1568, 768]);  view_228 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_24: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_150, view_229, permute_97);  primals_150 = None
    view_230: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_24, [8, 196, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_132: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_230);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_86: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_80, clone_132);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_133: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_133, getitem_35);  clone_133 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_17);  sub_53 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_46);  mul_85 = None
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_47);  mul_86 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1568, 768]" = torch.ops.aten.view.default(add_88, [1568, 768]);  add_88 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_25: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_152, view_231, permute_98);  primals_152 = None
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 196, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476)
    erf_8: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_89: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_89: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_89);  mul_87 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_134: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_134, [1568, 3072]);  clone_134 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_26: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_154, view_233, permute_99);  primals_154 = None
    view_234: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_26, [8, 196, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_135: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_234);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_90: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_86, clone_135);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_136: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_136, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_54: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_136, getitem_37);  clone_136 = None
    mul_90: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_18);  sub_54 = None
    mul_91: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_48);  mul_90 = None
    add_92: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_49);  mul_91 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    full_9: "f32[1, 196, 196, 3]" = torch.ops.aten.full.default([1, 196, 196, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_235: "i64[1, 14]" = torch.ops.aten.view.default(iota_18, [1, -1]);  iota_18 = None
    iota_19: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_236: "i64[14, 1]" = torch.ops.aten.view.default(iota_19, [-1, 1]);  iota_19 = None
    sub_55: "i64[14, 14]" = torch.ops.aten.sub.Tensor(view_235, view_236);  view_235 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    repeat_9: "i64[196, 196]" = torch.ops.aten.repeat.default(sub_55, [14, 14])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    unsqueeze_54: "i64[14, 1, 14]" = torch.ops.aten.unsqueeze.default(sub_55, 1);  sub_55 = None
    expand_64: "i64[14, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_54, [14, 14, 14]);  unsqueeze_54 = None
    clone_137: "i64[14, 14, 14]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_237: "i64[196, 14]" = torch.ops.aten.view.default(clone_137, [196, 14]);  clone_137 = None
    unsqueeze_55: "i64[196, 14, 1]" = torch.ops.aten.unsqueeze.default(view_237, 2);  view_237 = None
    expand_65: "i64[196, 14, 14]" = torch.ops.aten.expand.default(unsqueeze_55, [196, 14, 14]);  unsqueeze_55 = None
    clone_138: "i64[196, 14, 14]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_238: "i64[196, 196]" = torch.ops.aten.view.default(clone_138, [196, 196]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_19: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(repeat_9, 2)
    pow_20: "i64[196, 196]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 2)
    add_93: "i64[196, 196]" = torch.ops.aten.add.Tensor(pow_19, pow_20);  pow_19 = pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_56: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(add_93, 0);  add_93 = None
    slice_298: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_299: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_298, 1, 0, 9223372036854775807);  slice_298 = None
    slice_300: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_299, 2, 0, 9223372036854775807);  slice_299 = None
    select_90: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_300, 3, 2);  slice_300 = None
    copy_27: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_90, unsqueeze_56);  select_90 = unsqueeze_56 = None
    slice_301: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_302: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_301, 1, 0, 9223372036854775807)
    slice_303: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_302, 2, 0, 9223372036854775807)
    select_scatter_27: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_303, copy_27, 3, 2);  slice_303 = copy_27 = None
    slice_scatter_81: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_302, select_scatter_27, 2, 0, 9223372036854775807);  slice_302 = select_scatter_27 = None
    slice_scatter_82: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_301, slice_scatter_81, 1, 0, 9223372036854775807);  slice_301 = slice_scatter_81 = None
    slice_scatter_83: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_82, 0, 0, 9223372036854775807);  full_9 = slice_scatter_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_57: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(view_238, 0);  view_238 = None
    slice_310: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 0, 0, 9223372036854775807)
    slice_311: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_310, 1, 0, 9223372036854775807);  slice_310 = None
    slice_312: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_311, 2, 0, 9223372036854775807);  slice_311 = None
    select_93: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_312, 3, 1);  slice_312 = None
    copy_28: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_93, unsqueeze_57);  select_93 = unsqueeze_57 = None
    slice_313: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 0, 0, 9223372036854775807)
    slice_314: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_313, 1, 0, 9223372036854775807)
    slice_315: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_314, 2, 0, 9223372036854775807)
    select_scatter_28: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_315, copy_28, 3, 1);  slice_315 = copy_28 = None
    slice_scatter_84: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_314, select_scatter_28, 2, 0, 9223372036854775807);  slice_314 = select_scatter_28 = None
    slice_scatter_85: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_313, slice_scatter_84, 1, 0, 9223372036854775807);  slice_313 = slice_scatter_84 = None
    slice_scatter_86: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_83, slice_scatter_85, 0, 0, 9223372036854775807);  slice_scatter_83 = slice_scatter_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_58: "i64[1, 196, 196]" = torch.ops.aten.unsqueeze.default(repeat_9, 0);  repeat_9 = None
    slice_322: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 0, 0, 9223372036854775807)
    slice_323: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_322, 1, 0, 9223372036854775807);  slice_322 = None
    slice_324: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_323, 2, 0, 9223372036854775807);  slice_323 = None
    select_96: "f32[1, 196, 196]" = torch.ops.aten.select.int(slice_324, 3, 0);  slice_324 = None
    copy_29: "f32[1, 196, 196]" = torch.ops.aten.copy.default(select_96, unsqueeze_58);  select_96 = unsqueeze_58 = None
    slice_325: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 0, 0, 9223372036854775807)
    slice_326: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_325, 1, 0, 9223372036854775807)
    slice_327: "f32[1, 196, 196, 3]" = torch.ops.aten.slice.Tensor(slice_326, 2, 0, 9223372036854775807)
    select_scatter_29: "f32[1, 196, 196, 3]" = torch.ops.aten.select_scatter.default(slice_327, copy_29, 3, 0);  slice_327 = copy_29 = None
    slice_scatter_87: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_326, select_scatter_29, 2, 0, 9223372036854775807);  slice_326 = select_scatter_29 = None
    slice_scatter_88: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_325, slice_scatter_87, 1, 0, 9223372036854775807);  slice_325 = slice_scatter_87 = None
    slice_scatter_89: "f32[1, 196, 196, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_86, slice_scatter_88, 0, 0, 9223372036854775807);  slice_scatter_86 = slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    device_put_9: "f32[1, 196, 196, 3]" = torch.ops.prims.device_put.default(slice_scatter_89, device(type='cuda', index=0));  slice_scatter_89 = None
    convert_element_type_9: "f32[1, 196, 196, 3]" = torch.ops.prims.convert_element_type.default(device_put_9, torch.float32);  device_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_100: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    view_239: "f32[1568, 768]" = torch.ops.aten.view.default(add_92, [1568, 768])
    mm_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_239, permute_100)
    view_240: "f32[8, 196, 1536]" = torch.ops.aten.view.default(mm_27, [8, 196, 1536]);  mm_27 = None
    view_241: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.view.default(view_240, [8, 196, 2, 16, 48]);  view_240 = None
    permute_101: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_98: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 0)
    select_99: "f32[8, 16, 196, 48]" = torch.ops.aten.select.int(permute_101, 0, 1);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    expand_66: "f32[8, 196, 196, 3]" = torch.ops.aten.expand.default(convert_element_type_9, [8, -1, -1, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_102: "f32[3, 16]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    clone_139: "f32[8, 196, 196, 3]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_242: "f32[307328, 3]" = torch.ops.aten.view.default(clone_139, [307328, 3]);  clone_139 = None
    mm_28: "f32[307328, 16]" = torch.ops.aten.mm.default(view_242, permute_102);  permute_102 = None
    view_243: "f32[8, 196, 196, 16]" = torch.ops.aten.view.default(mm_28, [8, 196, 196, 16]);  mm_28 = None
    add_94: "f32[8, 196, 196, 16]" = torch.ops.aten.add.Tensor(view_243, primals_157);  view_243 = primals_157 = None
    permute_103: "f32[8, 16, 196, 196]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    permute_104: "f32[8, 16, 48, 196]" = torch.ops.aten.permute.default(select_99, [0, 1, 3, 2]);  select_99 = None
    expand_67: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 196, 48]);  select_98 = None
    clone_140: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_244: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_140, [128, 196, 48]);  clone_140 = None
    expand_68: "f32[8, 16, 48, 196]" = torch.ops.aten.expand.default(permute_104, [8, 16, 48, 196]);  permute_104 = None
    clone_141: "f32[8, 16, 48, 196]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_245: "f32[128, 48, 196]" = torch.ops.aten.view.default(clone_141, [128, 48, 196]);  clone_141 = None
    bmm_18: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 16, 196, 196]);  bmm_18 = None
    mul_92: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_246, 0.14433756729740643);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    amax_18: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(mul_92, [-1], True)
    sub_56: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_92, amax_18);  mul_92 = amax_18 = None
    exp_18: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_28: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_27: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_28);  exp_18 = sum_28 = None
    alias_36: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    clone_142: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    amax_19: "f32[8, 16, 196, 1]" = torch.ops.aten.amax.default(clone_142, [-1], True)
    sub_57: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(clone_142, amax_19);  clone_142 = amax_19 = None
    exp_19: "f32[8, 16, 196, 196]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_29: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_28: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_19, sum_29);  exp_19 = sum_29 = None
    alias_37: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(div_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_247: "f32[1, 16, 1, 1]" = torch.ops.aten.view.default(primals_50, [1, -1, 1, 1]);  primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247)
    alias_38: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_18)
    sub_58: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_18);  sigmoid_18 = None
    mul_93: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_58, div_27)
    sigmoid_19: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247);  view_247 = None
    alias_39: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    mul_94: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_19, div_28)
    add_95: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_30: "f32[8, 16, 196]" = torch.ops.aten.sum.dim_IntList(add_95, [-1])
    unsqueeze_59: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(sum_30, -1);  sum_30 = None
    clone_143: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(add_95)
    div_29: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_95, unsqueeze_59);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    clone_144: "f32[8, 16, 196, 196]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    view_248: "f32[1568, 768]" = torch.ops.aten.view.default(add_92, [1568, 768]);  add_92 = None
    mm_29: "f32[1568, 768]" = torch.ops.aten.mm.default(view_248, permute_105)
    view_249: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_29, [8, 196, 768]);  mm_29 = None
    view_250: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_249, [8, 196, 16, 48]);  view_249 = None
    permute_106: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_69: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(clone_144, [8, 16, 196, 196]);  clone_144 = None
    view_251: "f32[128, 196, 196]" = torch.ops.aten.view.default(expand_69, [128, 196, 196]);  expand_69 = None
    expand_70: "f32[8, 16, 196, 48]" = torch.ops.aten.expand.default(permute_106, [8, 16, 196, 48]);  permute_106 = None
    clone_145: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_252: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_145, [128, 196, 48]);  clone_145 = None
    bmm_19: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_19, [8, 16, 196, 48]);  bmm_19 = None
    permute_107: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_146: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_254: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_146, [8, 196, 768]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_255: "f32[1568, 768]" = torch.ops.aten.view.default(view_254, [1568, 768]);  view_254 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_27: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_160, view_255, permute_108);  primals_160 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_27, [8, 196, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    clone_147: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_256);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_96: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_90, clone_147);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_148: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_148, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_97: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_148, getitem_39);  clone_148 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_19);  sub_59 = None
    mul_96: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, primals_51);  mul_95 = None
    add_98: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_96, primals_52);  mul_96 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 768]" = torch.ops.aten.view.default(add_98, [1568, 768]);  add_98 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_28: "f32[1568, 3072]" = torch.ops.aten.addmm.default(primals_162, view_257, permute_109);  primals_162 = None
    view_258: "f32[8, 196, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 196, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_97: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476)
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_99: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_99: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_99);  mul_97 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_149: "f32[8, 196, 3072]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[1568, 3072]" = torch.ops.aten.view.default(clone_149, [1568, 3072]);  clone_149 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_29: "f32[1568, 768]" = torch.ops.aten.addmm.default(primals_164, view_259, permute_110);  primals_164 = None
    view_260: "f32[8, 196, 768]" = torch.ops.aten.view.default(addmm_29, [8, 196, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_150: "f32[8, 196, 768]" = torch.ops.aten.clone.default(view_260);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_100: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_96, clone_150);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, add_100], 1);  expand = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_101: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41)
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_20);  sub_60 = None
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, primals_53);  mul_100 = None
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_101, primals_54);  mul_101 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_261: "f32[1576, 768]" = torch.ops.aten.view.default(add_102, [1576, 768]);  add_102 = None
    mm_30: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_261, permute_111)
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.view.default(mm_30, [8, 197, 2304]);  mm_30 = None
    view_263: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.view.default(view_262, [8, 197, 3, 16, 48]);  view_262 = None
    permute_112: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_42: "f32[8, 16, 197, 48]" = unbind[0]
    getitem_43: "f32[8, 16, 197, 48]" = unbind[1]
    getitem_44: "f32[8, 16, 197, 48]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_113: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_71: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_42, [8, 16, 197, 48]);  getitem_42 = None
    clone_151: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_264: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_151, [128, 197, 48]);  clone_151 = None
    expand_72: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_113, [8, 16, 48, 197]);  permute_113 = None
    clone_152: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_265: "f32[128, 48, 197]" = torch.ops.aten.view.default(clone_152, [128, 48, 197]);  clone_152 = None
    bmm_20: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_264, view_265)
    view_266: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_20, [8, 16, 197, 197]);  bmm_20 = None
    mul_102: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_266, 0.14433756729740643);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_102, [-1], True)
    sub_61: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_102, amax_20);  mul_102 = amax_20 = None
    exp_20: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_31: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_30: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_20, sum_31);  exp_20 = sum_31 = None
    alias_40: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_153: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_73: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_153, [8, 16, 197, 197]);  clone_153 = None
    view_267: "f32[128, 197, 197]" = torch.ops.aten.view.default(expand_73, [128, 197, 197]);  expand_73 = None
    expand_74: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_44, [8, 16, 197, 48]);  getitem_44 = None
    clone_154: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_268: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_154, [128, 197, 48]);  clone_154 = None
    bmm_21: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_267, view_268)
    view_269: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_21, [8, 16, 197, 48]);  bmm_21 = None
    permute_114: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_155: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_270: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_155, [8, 197, 768]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_271: "f32[1576, 768]" = torch.ops.aten.view.default(view_270, [1576, 768]);  view_270 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_167, view_271, permute_115);  primals_167 = None
    view_272: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    clone_156: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_272);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_103: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, clone_156);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_45: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_46: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_104: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_46)
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_21);  sub_62 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_103, primals_55);  mul_103 = None
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_104, primals_56);  mul_104 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_273: "f32[1576, 768]" = torch.ops.aten.view.default(add_105, [1576, 768]);  add_105 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_31: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_169, view_273, permute_116);  primals_169 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 197, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_106);  mul_105 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_157: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_157, [1576, 3072]);  clone_157 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_171, view_275, permute_117);  primals_171 = None
    view_276: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_158: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_276);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_107: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_103, clone_158);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_108: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_48)
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_22);  sub_63 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, primals_57);  mul_108 = None
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_109, primals_58);  mul_109 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    view_277: "f32[1576, 768]" = torch.ops.aten.view.default(add_109, [1576, 768]);  add_109 = None
    mm_31: "f32[1576, 2304]" = torch.ops.aten.mm.default(view_277, permute_118)
    view_278: "f32[8, 197, 2304]" = torch.ops.aten.view.default(mm_31, [8, 197, 2304]);  mm_31 = None
    view_279: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.view.default(view_278, [8, 197, 3, 16, 48]);  view_278 = None
    permute_119: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.permute.default(view_279, [2, 0, 3, 1, 4]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_49: "f32[8, 16, 197, 48]" = unbind_1[0]
    getitem_50: "f32[8, 16, 197, 48]" = unbind_1[1]
    getitem_51: "f32[8, 16, 197, 48]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_120: "f32[8, 16, 48, 197]" = torch.ops.aten.permute.default(getitem_50, [0, 1, 3, 2]);  getitem_50 = None
    expand_75: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_49, [8, 16, 197, 48]);  getitem_49 = None
    clone_159: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_280: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_159, [128, 197, 48]);  clone_159 = None
    expand_76: "f32[8, 16, 48, 197]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 197]);  permute_120 = None
    clone_160: "f32[8, 16, 48, 197]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_281: "f32[128, 48, 197]" = torch.ops.aten.view.default(clone_160, [128, 48, 197]);  clone_160 = None
    bmm_22: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281)
    view_282: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_22, [8, 16, 197, 197]);  bmm_22 = None
    mul_110: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.14433756729740643);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 197, 1]" = torch.ops.aten.amax.default(mul_110, [-1], True)
    sub_64: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_110, amax_21);  mul_110 = amax_21 = None
    exp_21: "f32[8, 16, 197, 197]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_32: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_31: "f32[8, 16, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_32);  exp_21 = sum_32 = None
    alias_41: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    clone_161: "f32[8, 16, 197, 197]" = torch.ops.aten.clone.default(div_31);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_77: "f32[8, 16, 197, 197]" = torch.ops.aten.expand.default(clone_161, [8, 16, 197, 197]);  clone_161 = None
    view_283: "f32[128, 197, 197]" = torch.ops.aten.view.default(expand_77, [128, 197, 197]);  expand_77 = None
    expand_78: "f32[8, 16, 197, 48]" = torch.ops.aten.expand.default(getitem_51, [8, 16, 197, 48]);  getitem_51 = None
    clone_162: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_284: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_162, [128, 197, 48]);  clone_162 = None
    bmm_23: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_283, view_284)
    view_285: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_23, [8, 16, 197, 48]);  bmm_23 = None
    permute_121: "f32[8, 197, 16, 48]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_163: "f32[8, 197, 16, 48]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_163, [8, 197, 768]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.view.default(view_286, [1576, 768]);  view_286 = None
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_174, view_287, permute_122);  primals_174 = None
    view_288: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    clone_164: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_288);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_107, clone_164);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_53: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_111: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_53)
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_23);  sub_65 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_111, primals_59);  mul_111 = None
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_112, primals_60);  mul_112 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 768]" = torch.ops.aten.view.default(add_112, [1576, 768]);  add_112 = None
    permute_123: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_176, view_289, permute_123);  primals_176 = None
    view_290: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_113: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_113, add_113);  mul_113 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_165: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_165, [1576, 3072]);  clone_165 = None
    permute_124: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_178, view_291, permute_124);  primals_178 = None
    view_292: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_166: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_292);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, clone_166);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_115: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_55)
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_24);  sub_66 = None
    mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, primals_61);  mul_116 = None
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_117, primals_62);  mul_117 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_331: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_116, 0, 0, 9223372036854775807);  add_116 = None
    select_100: "f32[8, 768]" = torch.ops.aten.select.int(slice_331, 1, 0);  slice_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:375, code: x = self.head_drop(x)
    clone_167: "f32[8, 768]" = torch.ops.aten.clone.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    permute_125: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_180, clone_167, permute_125);  primals_180 = None
    permute_126: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_32: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_126);  permute_126 = None
    permute_127: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_33: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_127, clone_167);  permute_127 = clone_167 = None
    permute_128: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_33: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_293: "f32[1000]" = torch.ops.aten.view.default(sum_33, [1000]);  sum_33 = None
    permute_129: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_10: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_30: "f32[8, 197, 768]" = torch.ops.aten.select_scatter.default(full_10, mm_32, 1, 0);  full_10 = mm_32 = None
    full_11: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_90: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_11, select_scatter_30, 0, 0, 9223372036854775807);  full_11 = select_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_67: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_55);  add_114 = getitem_55 = None
    mul_118: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_24);  sub_67 = None
    mul_119: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_90, primals_61);  primals_61 = None
    mul_120: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_119, 768)
    sum_34: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True)
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_119, mul_118);  mul_119 = None
    sum_35: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_118, sum_35);  sum_35 = None
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_120, sum_34);  mul_120 = sum_34 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_122);  sub_68 = mul_122 = None
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_123: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_69);  div_32 = sub_69 = None
    mul_124: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_90, mul_118);  mul_118 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1]);  mul_124 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(slice_scatter_90, [0, 1]);  slice_scatter_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_294: "f32[1576, 768]" = torch.ops.aten.view.default(mul_123, [1576, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_294, permute_130);  permute_130 = None
    permute_131: "f32[768, 1576]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_131, view_291);  permute_131 = view_291 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    permute_133: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    view_296: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_125: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476)
    erf_12: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
    add_117: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_126: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_127: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, view_290)
    mul_128: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_127, -0.5);  mul_127 = None
    exp_22: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_128);  mul_128 = None
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_130: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, mul_129);  view_290 = mul_129 = None
    add_118: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_126, mul_130);  mul_126 = mul_130 = None
    mul_131: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_296, add_118);  view_296 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_297: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_131, [1576, 3072]);  mul_131 = None
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_297, permute_134);  permute_134 = None
    permute_135: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_135, view_289);  permute_135 = view_289 = None
    permute_136: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[3072]" = torch.ops.aten.view.default(sum_39, [3072]);  sum_39 = None
    permute_137: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_299: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_70: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_53);  add_110 = getitem_53 = None
    mul_132: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_23);  sub_70 = None
    mul_133: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_299, primals_59);  primals_59 = None
    mul_134: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_133, 768)
    sum_40: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)
    mul_135: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_133, mul_132);  mul_133 = None
    sum_41: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    mul_136: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_132, sum_41);  sum_41 = None
    sub_71: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_134, sum_40);  mul_134 = sum_40 = None
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_136);  sub_71 = mul_136 = None
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_137: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_72);  div_33 = sub_72 = None
    mul_138: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_299, mul_132);  mul_132 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1]);  mul_138 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_299, [0, 1]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_119: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_123, mul_137);  mul_123 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_300: "f32[1576, 768]" = torch.ops.aten.view.default(add_119, [1576, 768])
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_300, permute_138);  permute_138 = None
    permute_139: "f32[768, 1576]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_287);  permute_139 = view_287 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_302: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_303: "f32[8, 197, 16, 48]" = torch.ops.aten.view.default(view_302, [8, 197, 16, 48]);  view_302 = None
    permute_142: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    clone_168: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_304: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_168, [128, 197, 48]);  clone_168 = None
    permute_143: "f32[128, 197, 197]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    bmm_24: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(permute_143, view_304);  permute_143 = None
    permute_144: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    bmm_25: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_304, permute_144);  view_304 = permute_144 = None
    view_305: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_24, [8, 16, 197, 48]);  bmm_24 = None
    view_306: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_25, [8, 16, 197, 197]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    mul_139: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_306, alias_42);  view_306 = None
    sum_45: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [-1], True)
    mul_140: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(alias_42, sum_45);  alias_42 = sum_45 = None
    sub_73: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_141: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(sub_73, 0.14433756729740643);  sub_73 = None
    view_307: "f32[128, 197, 197]" = torch.ops.aten.view.default(mul_141, [128, 197, 197]);  mul_141 = None
    permute_145: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
    bmm_26: "f32[128, 48, 197]" = torch.ops.aten.bmm.default(permute_145, view_307);  permute_145 = None
    permute_146: "f32[128, 197, 48]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
    bmm_27: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_307, permute_146);  view_307 = permute_146 = None
    view_308: "f32[8, 16, 48, 197]" = torch.ops.aten.view.default(bmm_26, [8, 16, 48, 197]);  bmm_26 = None
    view_309: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_27, [8, 16, 197, 48]);  bmm_27 = None
    permute_147: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_308, [0, 1, 3, 2]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 16, 197, 48]" = torch.ops.aten.cat.default([view_309, permute_147, view_305]);  view_309 = permute_147 = view_305 = None
    view_310: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.view.default(cat_1, [3, 8, 16, 197, 48]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_148: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.permute.default(view_310, [1, 3, 0, 2, 4]);  view_310 = None
    clone_169: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_311: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_169, [8, 197, 2304]);  clone_169 = None
    view_312: "f32[1576, 2304]" = torch.ops.aten.view.default(view_311, [1576, 2304]);  view_311 = None
    permute_149: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_40: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_149, view_277);  permute_149 = view_277 = None
    permute_150: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    permute_151: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_41: "f32[1576, 768]" = torch.ops.aten.mm.default(view_312, permute_151);  view_312 = permute_151 = None
    view_313: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_41, [8, 197, 768]);  mm_41 = None
    permute_152: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_48);  add_107 = getitem_48 = None
    mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_22);  sub_74 = None
    mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_57);  primals_57 = None
    mul_144: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_46: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_142);  mul_143 = None
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_142, sum_47);  sum_47 = None
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_46);  mul_144 = sum_46 = None
    sub_76: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_146);  sub_75 = mul_146 = None
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_76);  div_34 = sub_76 = None
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_142);  mul_142 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_119, mul_147);  add_119 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_314: "f32[1576, 768]" = torch.ops.aten.view.default(add_120, [1576, 768])
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_314, permute_153);  permute_153 = None
    permute_154: "f32[768, 1576]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_154, view_275);  permute_154 = view_275 = None
    permute_155: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    view_316: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_149: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476)
    erf_13: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_149);  mul_149 = None
    add_121: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_150: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
    mul_151: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, view_274)
    mul_152: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_151, -0.5);  mul_151 = None
    exp_23: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_152);  mul_152 = None
    mul_153: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_154: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, mul_153);  view_274 = mul_153 = None
    add_122: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_150, mul_154);  mul_150 = mul_154 = None
    mul_155: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_316, add_122);  view_316 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_317: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_155, [1576, 3072]);  mul_155 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_317, permute_157);  permute_157 = None
    permute_158: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_158, view_273);  permute_158 = view_273 = None
    permute_159: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[3072]" = torch.ops.aten.view.default(sum_51, [3072]);  sum_51 = None
    permute_160: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_319: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_46);  add_103 = getitem_46 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_21);  sub_77 = None
    mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, primals_55);  primals_55 = None
    mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_157, 768)
    sum_52: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True)
    mul_159: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_157, mul_156);  mul_157 = None
    sum_53: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    mul_160: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_156, sum_53);  sum_53 = None
    sub_78: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_158, sum_52);  mul_158 = sum_52 = None
    sub_79: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_160);  sub_78 = mul_160 = None
    div_35: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_79);  div_35 = sub_79 = None
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, mul_156);  mul_156 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_123: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_120, mul_161);  add_120 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_320: "f32[1576, 768]" = torch.ops.aten.view.default(add_123, [1576, 768])
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_320, permute_161);  permute_161 = None
    permute_162: "f32[768, 1576]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_271);  permute_162 = view_271 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_322: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_323: "f32[8, 197, 16, 48]" = torch.ops.aten.view.default(view_322, [8, 197, 16, 48]);  view_322 = None
    permute_165: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
    clone_170: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    view_324: "f32[128, 197, 48]" = torch.ops.aten.view.default(clone_170, [128, 197, 48]);  clone_170 = None
    permute_166: "f32[128, 197, 197]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    bmm_28: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(permute_166, view_324);  permute_166 = None
    permute_167: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    bmm_29: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_324, permute_167);  view_324 = permute_167 = None
    view_325: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_28, [8, 16, 197, 48]);  bmm_28 = None
    view_326: "f32[8, 16, 197, 197]" = torch.ops.aten.view.default(bmm_29, [8, 16, 197, 197]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    alias_43: "f32[8, 16, 197, 197]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    mul_163: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_326, alias_43);  view_326 = None
    sum_57: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [-1], True)
    mul_164: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(alias_43, sum_57);  alias_43 = sum_57 = None
    sub_80: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_165: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(sub_80, 0.14433756729740643);  sub_80 = None
    view_327: "f32[128, 197, 197]" = torch.ops.aten.view.default(mul_165, [128, 197, 197]);  mul_165 = None
    permute_168: "f32[128, 48, 197]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    bmm_30: "f32[128, 48, 197]" = torch.ops.aten.bmm.default(permute_168, view_327);  permute_168 = None
    permute_169: "f32[128, 197, 48]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    bmm_31: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_327, permute_169);  view_327 = permute_169 = None
    view_328: "f32[8, 16, 48, 197]" = torch.ops.aten.view.default(bmm_30, [8, 16, 48, 197]);  bmm_30 = None
    view_329: "f32[8, 16, 197, 48]" = torch.ops.aten.view.default(bmm_31, [8, 16, 197, 48]);  bmm_31 = None
    permute_170: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_328, [0, 1, 3, 2]);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 16, 197, 48]" = torch.ops.aten.cat.default([view_329, permute_170, view_325]);  view_329 = permute_170 = view_325 = None
    view_330: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.view.default(cat_2, [3, 8, 16, 197, 48]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_171: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.permute.default(view_330, [1, 3, 0, 2, 4]);  view_330 = None
    clone_171: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_331: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_171, [8, 197, 2304]);  clone_171 = None
    view_332: "f32[1576, 2304]" = torch.ops.aten.view.default(view_331, [1576, 2304]);  view_331 = None
    permute_172: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_48: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_172, view_261);  permute_172 = view_261 = None
    permute_173: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    permute_174: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_49: "f32[1576, 768]" = torch.ops.aten.mm.default(view_332, permute_174);  view_332 = permute_174 = None
    view_333: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_49, [8, 197, 768]);  mm_49 = None
    permute_175: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41);  cat = getitem_41 = None
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_20);  sub_81 = None
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_333, primals_53);  primals_53 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
    sum_58: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_166);  mul_167 = None
    sum_59: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_166, sum_59);  sum_59 = None
    sub_82: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_58);  mul_168 = sum_58 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_170);  sub_82 = mul_170 = None
    div_36: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_83);  div_36 = sub_83 = None
    mul_172: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_333, mul_166);  mul_166 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_123, mul_171);  add_123 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_332: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_124, 1, 0, 1)
    slice_333: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_124, 1, 1, 197);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_172: "f32[8, 196, 768]" = torch.ops.aten.clone.default(slice_333, memory_format = torch.contiguous_format)
    view_334: "f32[1568, 768]" = torch.ops.aten.view.default(clone_172, [1568, 768]);  clone_172 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_50: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_334, permute_176);  permute_176 = None
    permute_177: "f32[768, 1568]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_259);  permute_177 = view_259 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_336: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_50, [8, 196, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_173: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476)
    erf_14: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_173);  mul_173 = None
    add_125: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_174: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_175: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, view_258)
    mul_176: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_175, -0.5);  mul_175 = None
    exp_24: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_176);  mul_176 = None
    mul_177: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_178: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, mul_177);  view_258 = mul_177 = None
    add_126: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_174, mul_178);  mul_174 = mul_178 = None
    mul_179: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_336, add_126);  view_336 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_337: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_179, [1568, 3072]);  mul_179 = None
    permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_52: "f32[1568, 768]" = torch.ops.aten.mm.default(view_337, permute_180);  permute_180 = None
    permute_181: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_337, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_257);  permute_181 = view_257 = None
    permute_182: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
    view_338: "f32[3072]" = torch.ops.aten.view.default(sum_63, [3072]);  sum_63 = None
    permute_183: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_339: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_52, [8, 196, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_173: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format);  add_96 = None
    sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_173, getitem_39);  clone_173 = getitem_39 = None
    mul_180: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_19);  sub_84 = None
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_51);  primals_51 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, 768)
    sum_64: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True)
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, mul_180);  mul_181 = None
    sum_65: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
    mul_184: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_180, sum_65);  sum_65 = None
    sub_85: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_182, sum_64);  mul_182 = sum_64 = None
    sub_86: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_184);  sub_85 = mul_184 = None
    div_37: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_185: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_86);  div_37 = sub_86 = None
    mul_186: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_180);  mul_180 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1]);  mul_186 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_127: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(slice_333, mul_185);  slice_333 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_340: "f32[1568, 768]" = torch.ops.aten.view.default(add_127, [1568, 768])
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_54: "f32[1568, 768]" = torch.ops.aten.mm.default(view_340, permute_184);  permute_184 = None
    permute_185: "f32[768, 1568]" = torch.ops.aten.permute.default(view_340, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_255);  permute_185 = view_255 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
    view_341: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_342: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_54, [8, 196, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_343: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_342, [8, 196, 16, 48]);  view_342 = None
    permute_188: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    clone_174: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_344: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_174, [128, 196, 48]);  clone_174 = None
    permute_189: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_32: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_189, view_344);  permute_189 = None
    permute_190: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_33: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_344, permute_190);  view_344 = permute_190 = None
    view_345: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_32, [8, 16, 196, 48]);  bmm_32 = None
    view_346: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_33, [8, 16, 196, 196]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_191: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_175: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    view_347: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_175, [8, 196, 768]);  clone_175 = None
    view_348: "f32[1568, 768]" = torch.ops.aten.view.default(view_347, [1568, 768]);  view_347 = None
    permute_192: "f32[768, 1568]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_192, view_248);  permute_192 = view_248 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_57: "f32[1568, 768]" = torch.ops.aten.mm.default(view_348, permute_194);  view_348 = permute_194 = None
    view_349: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_57, [8, 196, 768]);  mm_57 = None
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_38: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_143, unsqueeze_59);  clone_143 = None
    div_39: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_38, unsqueeze_59);  div_38 = None
    neg: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_346)
    mul_187: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg, div_39);  neg = div_39 = None
    div_40: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_346, unsqueeze_59);  view_346 = unsqueeze_59 = None
    sum_69: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [3], True);  mul_187 = None
    squeeze: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_69, -1);  sum_69 = None
    unsqueeze_60: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
    expand_79: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_60, [8, 16, 196, 196]);  unsqueeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_128: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_40, expand_79);  div_40 = expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_188: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_19);  sigmoid_19 = None
    mul_189: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, div_28);  div_28 = None
    sum_70: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3], True);  mul_189 = None
    alias_44: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    sub_87: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_44)
    mul_190: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_44, sub_87);  alias_44 = sub_87 = None
    mul_191: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_70, mul_190);  sum_70 = mul_190 = None
    mul_192: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, sub_58);  sub_58 = None
    mul_193: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, div_27);  add_128 = div_27 = None
    sum_71: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 2, 3], True);  mul_193 = None
    neg_1: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_71);  sum_71 = None
    alias_45: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    sub_88: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_45)
    mul_194: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_45, sub_88);  alias_45 = sub_88 = None
    mul_195: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_1, mul_194);  neg_1 = mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_129: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_350: "f32[16]" = torch.ops.aten.view.default(add_129, [16]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_46: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    mul_196: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_188, alias_46);  mul_188 = None
    sum_72: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [-1], True)
    mul_197: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_46, sum_72);  alias_46 = sum_72 = None
    sub_89: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_47: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_198: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_192, alias_47);  mul_192 = None
    sum_73: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [-1], True)
    mul_199: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_47, sum_73);  alias_47 = sum_73 = None
    sub_90: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_200: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_90, 0.14433756729740643);  sub_90 = None
    view_351: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_200, [128, 196, 196]);  mul_200 = None
    permute_196: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    bmm_34: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_196, view_351);  permute_196 = None
    permute_197: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    bmm_35: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_351, permute_197);  view_351 = permute_197 = None
    view_352: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_34, [8, 16, 48, 196]);  bmm_34 = None
    view_353: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_35, [8, 16, 196, 48]);  bmm_35 = None
    permute_198: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_352, [0, 1, 3, 2]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_199: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_89, [0, 2, 3, 1]);  sub_89 = None
    sum_74: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_199, [0, 1, 2], True)
    view_354: "f32[16]" = torch.ops.aten.view.default(sum_74, [16]);  sum_74 = None
    clone_176: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_355: "f32[307328, 16]" = torch.ops.aten.view.default(clone_176, [307328, 16]);  clone_176 = None
    permute_200: "f32[16, 307328]" = torch.ops.aten.permute.default(view_355, [1, 0]);  view_355 = None
    mm_58: "f32[16, 3]" = torch.ops.aten.mm.default(permute_200, view_242);  permute_200 = view_242 = None
    permute_201: "f32[3, 16]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    permute_202: "f32[16, 3]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_12: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_31: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_12, permute_198, 0, 1);  full_12 = permute_198 = None
    full_13: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_32: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_13, view_353, 0, 0);  full_13 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_130: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_31, select_scatter_32);  select_scatter_31 = select_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_203: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_130, [1, 3, 0, 2, 4]);  add_130 = None
    clone_177: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_356: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_177, [8, 196, 1536]);  clone_177 = None
    view_357: "f32[1568, 1536]" = torch.ops.aten.view.default(view_356, [1568, 1536]);  view_356 = None
    permute_204: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_59: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_204, view_239);  permute_204 = view_239 = None
    permute_205: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    permute_206: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_60: "f32[1568, 768]" = torch.ops.aten.mm.default(view_357, permute_206);  view_357 = permute_206 = None
    view_358: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_60, [8, 196, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_131: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_349, view_358);  view_349 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_207: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_178: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format);  add_90 = None
    sub_91: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_178, getitem_37);  clone_178 = getitem_37 = None
    mul_201: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_18);  sub_91 = None
    mul_202: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_48);  primals_48 = None
    mul_203: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_202, 768)
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
    mul_204: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_201);  mul_202 = None
    sum_76: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_201, sum_76);  sum_76 = None
    sub_92: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_203, sum_75);  mul_203 = sum_75 = None
    sub_93: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_205);  sub_92 = mul_205 = None
    div_41: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_206: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_93);  div_41 = sub_93 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_201);  mul_201 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_127, mul_206);  add_127 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[1568, 768]" = torch.ops.aten.view.default(add_132, [1568, 768])
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_61: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_359, permute_208);  permute_208 = None
    permute_209: "f32[768, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_233);  permute_209 = view_233 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_361: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_61, [8, 196, 3072]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476)
    erf_15: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_133: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_209: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_133, 0.5);  add_133 = None
    mul_210: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, view_232)
    mul_211: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_210, -0.5);  mul_210 = None
    exp_25: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_211);  mul_211 = None
    mul_212: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_213: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, mul_212);  view_232 = mul_212 = None
    add_134: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_209, mul_213);  mul_209 = mul_213 = None
    mul_214: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_361, add_134);  view_361 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_362: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_214, [1568, 3072]);  mul_214 = None
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_63: "f32[1568, 768]" = torch.ops.aten.mm.default(view_362, permute_212);  permute_212 = None
    permute_213: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_64: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_231);  permute_213 = view_231 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[3072]" = torch.ops.aten.view.default(sum_80, [3072]);  sum_80 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_364: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_63, [8, 196, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_179: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format);  add_86 = None
    sub_94: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_179, getitem_35);  clone_179 = getitem_35 = None
    mul_215: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_17);  sub_94 = None
    mul_216: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, primals_46);  primals_46 = None
    mul_217: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_216, 768)
    sum_81: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
    mul_218: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_215);  mul_216 = None
    sum_82: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
    mul_219: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_215, sum_82);  sum_82 = None
    sub_95: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_217, sum_81);  mul_217 = sum_81 = None
    sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_219);  sub_95 = mul_219 = None
    div_42: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_220: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_96);  div_42 = sub_96 = None
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, mul_215);  mul_215 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 1]);  mul_221 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_364, [0, 1]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_135: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_132, mul_220);  add_132 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_365: "f32[1568, 768]" = torch.ops.aten.view.default(add_135, [1568, 768])
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_65: "f32[1568, 768]" = torch.ops.aten.mm.default(view_365, permute_216);  permute_216 = None
    permute_217: "f32[768, 1568]" = torch.ops.aten.permute.default(view_365, [1, 0])
    mm_66: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_229);  permute_217 = view_229 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
    view_366: "f32[768]" = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_367: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_65, [8, 196, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_368: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_367, [8, 196, 16, 48]);  view_367 = None
    permute_220: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_180: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    view_369: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_180, [128, 196, 48]);  clone_180 = None
    permute_221: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    bmm_36: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_221, view_369);  permute_221 = None
    permute_222: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
    bmm_37: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_369, permute_222);  view_369 = permute_222 = None
    view_370: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_36, [8, 16, 196, 48]);  bmm_36 = None
    view_371: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_37, [8, 16, 196, 196]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_223: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    clone_181: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    view_372: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_181, [8, 196, 768]);  clone_181 = None
    view_373: "f32[1568, 768]" = torch.ops.aten.view.default(view_372, [1568, 768]);  view_372 = None
    permute_224: "f32[768, 1568]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_224, view_222);  permute_224 = view_222 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_68: "f32[1568, 768]" = torch.ops.aten.mm.default(view_373, permute_226);  view_373 = permute_226 = None
    view_374: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_68, [8, 196, 768]);  mm_68 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_43: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_128, unsqueeze_53);  clone_128 = None
    div_44: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_43, unsqueeze_53);  div_43 = None
    neg_2: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_371)
    mul_222: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_2, div_44);  neg_2 = div_44 = None
    div_45: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_371, unsqueeze_53);  view_371 = unsqueeze_53 = None
    sum_86: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [3], True);  mul_222 = None
    squeeze_1: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_86, -1);  sum_86 = None
    unsqueeze_61: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_1, -1);  squeeze_1 = None
    expand_80: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_61, [8, 16, 196, 196]);  unsqueeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_136: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_45, expand_80);  div_45 = expand_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_223: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, sigmoid_17);  sigmoid_17 = None
    mul_224: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, div_25);  div_25 = None
    sum_87: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 2, 3], True);  mul_224 = None
    alias_48: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    sub_97: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_48)
    mul_225: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_48, sub_97);  alias_48 = sub_97 = None
    mul_226: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_225);  sum_87 = mul_225 = None
    mul_227: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, sub_52);  sub_52 = None
    mul_228: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, div_24);  add_136 = div_24 = None
    sum_88: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 2, 3], True);  mul_228 = None
    neg_3: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_88);  sum_88 = None
    alias_49: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    sub_98: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_49)
    mul_229: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_49, sub_98);  alias_49 = sub_98 = None
    mul_230: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_3, mul_229);  neg_3 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_137: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_226, mul_230);  mul_226 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_375: "f32[16]" = torch.ops.aten.view.default(add_137, [16]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_50: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_231: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_223, alias_50);  mul_223 = None
    sum_89: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [-1], True)
    mul_232: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_50, sum_89);  alias_50 = sum_89 = None
    sub_99: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_51: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_233: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_227, alias_51);  mul_227 = None
    sum_90: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [-1], True)
    mul_234: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_51, sum_90);  alias_51 = sum_90 = None
    sub_100: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_235: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_100, 0.14433756729740643);  sub_100 = None
    view_376: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_235, [128, 196, 196]);  mul_235 = None
    permute_228: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
    bmm_38: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_228, view_376);  permute_228 = None
    permute_229: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    bmm_39: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_376, permute_229);  view_376 = permute_229 = None
    view_377: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_38, [8, 16, 48, 196]);  bmm_38 = None
    view_378: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_39, [8, 16, 196, 48]);  bmm_39 = None
    permute_230: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_377, [0, 1, 3, 2]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_231: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_99, [0, 2, 3, 1]);  sub_99 = None
    sum_91: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_231, [0, 1, 2], True)
    view_379: "f32[16]" = torch.ops.aten.view.default(sum_91, [16]);  sum_91 = None
    clone_182: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_380: "f32[307328, 16]" = torch.ops.aten.view.default(clone_182, [307328, 16]);  clone_182 = None
    permute_232: "f32[16, 307328]" = torch.ops.aten.permute.default(view_380, [1, 0]);  view_380 = None
    mm_69: "f32[16, 3]" = torch.ops.aten.mm.default(permute_232, view_216);  permute_232 = view_216 = None
    permute_233: "f32[3, 16]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    permute_234: "f32[16, 3]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_14: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_33: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_14, permute_230, 0, 1);  full_14 = permute_230 = None
    full_15: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_34: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_15, view_378, 0, 0);  full_15 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_138: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_33, select_scatter_34);  select_scatter_33 = select_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_235: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_138, [1, 3, 0, 2, 4]);  add_138 = None
    clone_183: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_381: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_183, [8, 196, 1536]);  clone_183 = None
    view_382: "f32[1568, 1536]" = torch.ops.aten.view.default(view_381, [1568, 1536]);  view_381 = None
    permute_236: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_382, [1, 0])
    mm_70: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_236, view_213);  permute_236 = view_213 = None
    permute_237: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    permute_238: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_71: "f32[1568, 768]" = torch.ops.aten.mm.default(view_382, permute_238);  view_382 = permute_238 = None
    view_383: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_71, [8, 196, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_139: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_374, view_383);  view_374 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_239: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_184: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format);  add_80 = None
    sub_101: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_184, getitem_33);  clone_184 = getitem_33 = None
    mul_236: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_16);  sub_101 = None
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_139, primals_43);  primals_43 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_92: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_236);  mul_237 = None
    sum_93: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_236, sum_93);  sum_93 = None
    sub_102: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_92);  mul_238 = sum_92 = None
    sub_103: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_240);  sub_102 = mul_240 = None
    div_46: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_241: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_103);  div_46 = sub_103 = None
    mul_242: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_139, mul_236);  mul_236 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_135, mul_241);  add_135 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_384: "f32[1568, 768]" = torch.ops.aten.view.default(add_140, [1568, 768])
    permute_240: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_72: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_384, permute_240);  permute_240 = None
    permute_241: "f32[768, 1568]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_207);  permute_241 = view_207 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_386: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_72, [8, 196, 3072]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_243: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_16: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
    add_141: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_244: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_141, 0.5);  add_141 = None
    mul_245: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, view_206)
    mul_246: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_245, -0.5);  mul_245 = None
    exp_26: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_246);  mul_246 = None
    mul_247: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_248: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, mul_247);  view_206 = mul_247 = None
    add_142: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_244, mul_248);  mul_244 = mul_248 = None
    mul_249: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_386, add_142);  view_386 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_387: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_249, [1568, 3072]);  mul_249 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_74: "f32[1568, 768]" = torch.ops.aten.mm.default(view_387, permute_244);  permute_244 = None
    permute_245: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_205);  permute_245 = view_205 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[3072]" = torch.ops.aten.view.default(sum_97, [3072]);  sum_97 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_389: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_74, [8, 196, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_185: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format);  add_76 = None
    sub_104: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_185, getitem_31);  clone_185 = getitem_31 = None
    mul_250: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_15);  sub_104 = None
    mul_251: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_389, primals_41);  primals_41 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
    sum_98: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_250);  mul_251 = None
    sum_99: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_250, sum_99);  sum_99 = None
    sub_105: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_98);  mul_252 = sum_98 = None
    sub_106: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_254);  sub_105 = mul_254 = None
    div_47: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_255: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_106);  div_47 = sub_106 = None
    mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_389, mul_250);  mul_250 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_389, [0, 1]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_143: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_140, mul_255);  add_140 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_390: "f32[1568, 768]" = torch.ops.aten.view.default(add_143, [1568, 768])
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_76: "f32[1568, 768]" = torch.ops.aten.mm.default(view_390, permute_248);  permute_248 = None
    permute_249: "f32[768, 1568]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_249, view_203);  permute_249 = view_203 = None
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_392: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_76, [8, 196, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_393: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_392, [8, 196, 16, 48]);  view_392 = None
    permute_252: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_186: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
    view_394: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_186, [128, 196, 48]);  clone_186 = None
    permute_253: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    bmm_40: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_253, view_394);  permute_253 = None
    permute_254: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
    bmm_41: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_394, permute_254);  view_394 = permute_254 = None
    view_395: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_40, [8, 16, 196, 48]);  bmm_40 = None
    view_396: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_41, [8, 16, 196, 196]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_255: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
    clone_187: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_397: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_187, [8, 196, 768]);  clone_187 = None
    view_398: "f32[1568, 768]" = torch.ops.aten.view.default(view_397, [1568, 768]);  view_397 = None
    permute_256: "f32[768, 1568]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_78: "f32[768, 768]" = torch.ops.aten.mm.default(permute_256, view_196);  permute_256 = view_196 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_79: "f32[1568, 768]" = torch.ops.aten.mm.default(view_398, permute_258);  view_398 = permute_258 = None
    view_399: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_79, [8, 196, 768]);  mm_79 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_48: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_113, unsqueeze_47);  clone_113 = None
    div_49: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_48, unsqueeze_47);  div_48 = None
    neg_4: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_396)
    mul_257: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_4, div_49);  neg_4 = div_49 = None
    div_50: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_396, unsqueeze_47);  view_396 = unsqueeze_47 = None
    sum_103: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [3], True);  mul_257 = None
    squeeze_2: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_103, -1);  sum_103 = None
    unsqueeze_62: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_2, -1);  squeeze_2 = None
    expand_81: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_62, [8, 16, 196, 196]);  unsqueeze_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_144: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_50, expand_81);  div_50 = expand_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_258: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, sigmoid_15);  sigmoid_15 = None
    mul_259: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, div_22);  div_22 = None
    sum_104: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 2, 3], True);  mul_259 = None
    alias_52: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    sub_107: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_52)
    mul_260: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_52, sub_107);  alias_52 = sub_107 = None
    mul_261: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_260);  sum_104 = mul_260 = None
    mul_262: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, sub_46);  sub_46 = None
    mul_263: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, div_21);  add_144 = div_21 = None
    sum_105: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3], True);  mul_263 = None
    neg_5: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_105);  sum_105 = None
    alias_53: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    sub_108: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_53)
    mul_264: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_53, sub_108);  alias_53 = sub_108 = None
    mul_265: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_5, mul_264);  neg_5 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_145: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_400: "f32[16]" = torch.ops.aten.view.default(add_145, [16]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_54: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_266: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_258, alias_54);  mul_258 = None
    sum_106: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [-1], True)
    mul_267: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_54, sum_106);  alias_54 = sum_106 = None
    sub_109: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_55: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    mul_268: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_262, alias_55);  mul_262 = None
    sum_107: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [-1], True)
    mul_269: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_55, sum_107);  alias_55 = sum_107 = None
    sub_110: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_270: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_110, 0.14433756729740643);  sub_110 = None
    view_401: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_270, [128, 196, 196]);  mul_270 = None
    permute_260: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    bmm_42: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_260, view_401);  permute_260 = None
    permute_261: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    bmm_43: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_401, permute_261);  view_401 = permute_261 = None
    view_402: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_42, [8, 16, 48, 196]);  bmm_42 = None
    view_403: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_43, [8, 16, 196, 48]);  bmm_43 = None
    permute_262: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_263: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_109, [0, 2, 3, 1]);  sub_109 = None
    sum_108: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_263, [0, 1, 2], True)
    view_404: "f32[16]" = torch.ops.aten.view.default(sum_108, [16]);  sum_108 = None
    clone_188: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    view_405: "f32[307328, 16]" = torch.ops.aten.view.default(clone_188, [307328, 16]);  clone_188 = None
    permute_264: "f32[16, 307328]" = torch.ops.aten.permute.default(view_405, [1, 0]);  view_405 = None
    mm_80: "f32[16, 3]" = torch.ops.aten.mm.default(permute_264, view_190);  permute_264 = view_190 = None
    permute_265: "f32[3, 16]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    permute_266: "f32[16, 3]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_16: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_16, permute_262, 0, 1);  full_16 = permute_262 = None
    full_17: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_36: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_17, view_403, 0, 0);  full_17 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_146: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_35, select_scatter_36);  select_scatter_35 = select_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_267: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_146, [1, 3, 0, 2, 4]);  add_146 = None
    clone_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_406: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_189, [8, 196, 1536]);  clone_189 = None
    view_407: "f32[1568, 1536]" = torch.ops.aten.view.default(view_406, [1568, 1536]);  view_406 = None
    permute_268: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_81: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_268, view_187);  permute_268 = view_187 = None
    permute_269: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    permute_270: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_82: "f32[1568, 768]" = torch.ops.aten.mm.default(view_407, permute_270);  view_407 = permute_270 = None
    view_408: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_82, [8, 196, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_147: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_399, view_408);  view_399 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_271: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_190: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format);  add_70 = None
    sub_111: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_190, getitem_29);  clone_190 = getitem_29 = None
    mul_271: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_14);  sub_111 = None
    mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_147, primals_38);  primals_38 = None
    mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_109: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_271);  mul_272 = None
    sum_110: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_271, sum_110);  sum_110 = None
    sub_112: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_109);  mul_273 = sum_109 = None
    sub_113: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_275);  sub_112 = mul_275 = None
    div_51: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_276: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_113);  div_51 = sub_113 = None
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_271);  mul_271 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_148: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_143, mul_276);  add_143 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_409: "f32[1568, 768]" = torch.ops.aten.view.default(add_148, [1568, 768])
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_83: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_409, permute_272);  permute_272 = None
    permute_273: "f32[768, 1568]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_84: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_273, view_181);  permute_273 = view_181 = None
    permute_274: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_411: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_83, [8, 196, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_278: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476)
    erf_17: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
    add_149: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_279: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_149, 0.5);  add_149 = None
    mul_280: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, view_180)
    mul_281: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_280, -0.5);  mul_280 = None
    exp_27: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_281);  mul_281 = None
    mul_282: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_283: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, mul_282);  view_180 = mul_282 = None
    add_150: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_279, mul_283);  mul_279 = mul_283 = None
    mul_284: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_411, add_150);  view_411 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_412: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_284, [1568, 3072]);  mul_284 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_85: "f32[1568, 768]" = torch.ops.aten.mm.default(view_412, permute_276);  permute_276 = None
    permute_277: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_86: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_277, view_179);  permute_277 = view_179 = None
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
    view_413: "f32[3072]" = torch.ops.aten.view.default(sum_114, [3072]);  sum_114 = None
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_414: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_85, [8, 196, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_191: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format);  add_66 = None
    sub_114: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_191, getitem_27);  clone_191 = getitem_27 = None
    mul_285: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_13);  sub_114 = None
    mul_286: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_414, primals_36);  primals_36 = None
    mul_287: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_286, 768)
    sum_115: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_285);  mul_286 = None
    sum_116: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_285, sum_116);  sum_116 = None
    sub_115: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_287, sum_115);  mul_287 = sum_115 = None
    sub_116: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_289);  sub_115 = mul_289 = None
    div_52: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_290: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_116);  div_52 = sub_116 = None
    mul_291: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_414, mul_285);  mul_285 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_414, [0, 1]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_148, mul_290);  add_148 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_415: "f32[1568, 768]" = torch.ops.aten.view.default(add_151, [1568, 768])
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_87: "f32[1568, 768]" = torch.ops.aten.mm.default(view_415, permute_280);  permute_280 = None
    permute_281: "f32[768, 1568]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_281, view_177);  permute_281 = view_177 = None
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_417: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_87, [8, 196, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_418: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_417, [8, 196, 16, 48]);  view_417 = None
    permute_284: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    clone_192: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
    view_419: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_192, [128, 196, 48]);  clone_192 = None
    permute_285: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_44: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_285, view_419);  permute_285 = None
    permute_286: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    bmm_45: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_419, permute_286);  view_419 = permute_286 = None
    view_420: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_44, [8, 16, 196, 48]);  bmm_44 = None
    view_421: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_45, [8, 16, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_287: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    clone_193: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_422: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_193, [8, 196, 768]);  clone_193 = None
    view_423: "f32[1568, 768]" = torch.ops.aten.view.default(view_422, [1568, 768]);  view_422 = None
    permute_288: "f32[768, 1568]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_288, view_170);  permute_288 = view_170 = None
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_90: "f32[1568, 768]" = torch.ops.aten.mm.default(view_423, permute_290);  view_423 = permute_290 = None
    view_424: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_90, [8, 196, 768]);  mm_90 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_53: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_98, unsqueeze_41);  clone_98 = None
    div_54: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_53, unsqueeze_41);  div_53 = None
    neg_6: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_421)
    mul_292: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_6, div_54);  neg_6 = div_54 = None
    div_55: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_421, unsqueeze_41);  view_421 = unsqueeze_41 = None
    sum_120: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [3], True);  mul_292 = None
    squeeze_3: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_120, -1);  sum_120 = None
    unsqueeze_63: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_3, -1);  squeeze_3 = None
    expand_82: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_63, [8, 16, 196, 196]);  unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_152: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_55, expand_82);  div_55 = expand_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_293: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, sigmoid_13);  sigmoid_13 = None
    mul_294: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, div_19);  div_19 = None
    sum_121: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3], True);  mul_294 = None
    alias_56: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    sub_117: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_56)
    mul_295: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_56, sub_117);  alias_56 = sub_117 = None
    mul_296: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_121, mul_295);  sum_121 = mul_295 = None
    mul_297: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, sub_40);  sub_40 = None
    mul_298: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, div_18);  add_152 = div_18 = None
    sum_122: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3], True);  mul_298 = None
    neg_7: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_122);  sum_122 = None
    alias_57: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    sub_118: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_57)
    mul_299: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_57, sub_118);  alias_57 = sub_118 = None
    mul_300: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_7, mul_299);  neg_7 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_153: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_296, mul_300);  mul_296 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_425: "f32[16]" = torch.ops.aten.view.default(add_153, [16]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_58: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    mul_301: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_293, alias_58);  mul_293 = None
    sum_123: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_58, sum_123);  alias_58 = sum_123 = None
    sub_119: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_59: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_303: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_297, alias_59);  mul_297 = None
    sum_124: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [-1], True)
    mul_304: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_59, sum_124);  alias_59 = sum_124 = None
    sub_120: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_305: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_120, 0.14433756729740643);  sub_120 = None
    view_426: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_305, [128, 196, 196]);  mul_305 = None
    permute_292: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_46: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_292, view_426);  permute_292 = None
    permute_293: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_47: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_426, permute_293);  view_426 = permute_293 = None
    view_427: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_46, [8, 16, 48, 196]);  bmm_46 = None
    view_428: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_47, [8, 16, 196, 48]);  bmm_47 = None
    permute_294: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_427, [0, 1, 3, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_295: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_119, [0, 2, 3, 1]);  sub_119 = None
    sum_125: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_295, [0, 1, 2], True)
    view_429: "f32[16]" = torch.ops.aten.view.default(sum_125, [16]);  sum_125 = None
    clone_194: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_430: "f32[307328, 16]" = torch.ops.aten.view.default(clone_194, [307328, 16]);  clone_194 = None
    permute_296: "f32[16, 307328]" = torch.ops.aten.permute.default(view_430, [1, 0]);  view_430 = None
    mm_91: "f32[16, 3]" = torch.ops.aten.mm.default(permute_296, view_164);  permute_296 = view_164 = None
    permute_297: "f32[3, 16]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    permute_298: "f32[16, 3]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_18: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_37: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_18, permute_294, 0, 1);  full_18 = permute_294 = None
    full_19: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_38: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_19, view_428, 0, 0);  full_19 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_154: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_37, select_scatter_38);  select_scatter_37 = select_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_299: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_154, [1, 3, 0, 2, 4]);  add_154 = None
    clone_195: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_431: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_195, [8, 196, 1536]);  clone_195 = None
    view_432: "f32[1568, 1536]" = torch.ops.aten.view.default(view_431, [1568, 1536]);  view_431 = None
    permute_300: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_92: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_300, view_161);  permute_300 = view_161 = None
    permute_301: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    permute_302: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_93: "f32[1568, 768]" = torch.ops.aten.mm.default(view_432, permute_302);  view_432 = permute_302 = None
    view_433: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_93, [8, 196, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_155: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_424, view_433);  view_424 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_303: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_196: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format);  add_60 = None
    sub_121: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_196, getitem_25);  clone_196 = getitem_25 = None
    mul_306: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_12);  sub_121 = None
    mul_307: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_33);  primals_33 = None
    mul_308: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_307, 768)
    sum_126: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
    mul_309: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_307, mul_306);  mul_307 = None
    sum_127: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
    mul_310: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_306, sum_127);  sum_127 = None
    sub_122: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_308, sum_126);  mul_308 = sum_126 = None
    sub_123: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_310);  sub_122 = mul_310 = None
    div_56: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_311: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_123);  div_56 = sub_123 = None
    mul_312: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_306);  mul_306 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_156: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_151, mul_311);  add_151 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[1568, 768]" = torch.ops.aten.view.default(add_156, [1568, 768])
    permute_304: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_94: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_434, permute_304);  permute_304 = None
    permute_305: "f32[768, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_95: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_305, view_155);  permute_305 = view_155 = None
    permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_436: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_94, [8, 196, 3072]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_313: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476)
    erf_18: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
    add_157: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_314: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_315: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, view_154)
    mul_316: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_315, -0.5);  mul_315 = None
    exp_28: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_316);  mul_316 = None
    mul_317: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_318: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, mul_317);  view_154 = mul_317 = None
    add_158: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_314, mul_318);  mul_314 = mul_318 = None
    mul_319: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_436, add_158);  view_436 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_319, [1568, 3072]);  mul_319 = None
    permute_308: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_96: "f32[1568, 768]" = torch.ops.aten.mm.default(view_437, permute_308);  permute_308 = None
    permute_309: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_97: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_309, view_153);  permute_309 = view_153 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_131: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[3072]" = torch.ops.aten.view.default(sum_131, [3072]);  sum_131 = None
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_439: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_96, [8, 196, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_197: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format);  add_56 = None
    sub_124: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_197, getitem_23);  clone_197 = getitem_23 = None
    mul_320: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_11);  sub_124 = None
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_439, primals_31);  primals_31 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, 768)
    sum_132: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, mul_320);  mul_321 = None
    sum_133: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_320, sum_133);  sum_133 = None
    sub_125: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_322, sum_132);  mul_322 = sum_132 = None
    sub_126: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_324);  sub_125 = mul_324 = None
    div_57: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_325: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_126);  div_57 = sub_126 = None
    mul_326: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_439, mul_320);  mul_320 = None
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_439, [0, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_159: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_156, mul_325);  add_156 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_440: "f32[1568, 768]" = torch.ops.aten.view.default(add_159, [1568, 768])
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_98: "f32[1568, 768]" = torch.ops.aten.mm.default(view_440, permute_312);  permute_312 = None
    permute_313: "f32[768, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_313, view_151);  permute_313 = view_151 = None
    permute_314: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_442: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_98, [8, 196, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_443: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_442, [8, 196, 16, 48]);  view_442 = None
    permute_316: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    clone_198: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_444: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_198, [128, 196, 48]);  clone_198 = None
    permute_317: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_48: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_317, view_444);  permute_317 = None
    permute_318: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
    bmm_49: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_444, permute_318);  view_444 = permute_318 = None
    view_445: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_48, [8, 16, 196, 48]);  bmm_48 = None
    view_446: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_49, [8, 16, 196, 196]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_319: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_199: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_447: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_199, [8, 196, 768]);  clone_199 = None
    view_448: "f32[1568, 768]" = torch.ops.aten.view.default(view_447, [1568, 768]);  view_447 = None
    permute_320: "f32[768, 1568]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_100: "f32[768, 768]" = torch.ops.aten.mm.default(permute_320, view_144);  permute_320 = view_144 = None
    permute_321: "f32[768, 768]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_101: "f32[1568, 768]" = torch.ops.aten.mm.default(view_448, permute_322);  view_448 = permute_322 = None
    view_449: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_101, [8, 196, 768]);  mm_101 = None
    permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_58: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_83, unsqueeze_35);  clone_83 = None
    div_59: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_58, unsqueeze_35);  div_58 = None
    neg_8: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_446)
    mul_327: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_8, div_59);  neg_8 = div_59 = None
    div_60: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_446, unsqueeze_35);  view_446 = unsqueeze_35 = None
    sum_137: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [3], True);  mul_327 = None
    squeeze_4: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_137, -1);  sum_137 = None
    unsqueeze_64: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_4, -1);  squeeze_4 = None
    expand_83: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_64, [8, 16, 196, 196]);  unsqueeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_160: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_60, expand_83);  div_60 = expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_328: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, sigmoid_11);  sigmoid_11 = None
    mul_329: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, div_16);  div_16 = None
    sum_138: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3], True);  mul_329 = None
    alias_60: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    sub_127: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_60)
    mul_330: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_60, sub_127);  alias_60 = sub_127 = None
    mul_331: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_138, mul_330);  sum_138 = mul_330 = None
    mul_332: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, sub_34);  sub_34 = None
    mul_333: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, div_15);  add_160 = div_15 = None
    sum_139: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3], True);  mul_333 = None
    neg_9: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_139);  sum_139 = None
    alias_61: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    sub_128: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_61)
    mul_334: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_61, sub_128);  alias_61 = sub_128 = None
    mul_335: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_9, mul_334);  neg_9 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_161: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_331, mul_335);  mul_331 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_450: "f32[16]" = torch.ops.aten.view.default(add_161, [16]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_62: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_336: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_328, alias_62);  mul_328 = None
    sum_140: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [-1], True)
    mul_337: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_62, sum_140);  alias_62 = sum_140 = None
    sub_129: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_63: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_338: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_332, alias_63);  mul_332 = None
    sum_141: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [-1], True)
    mul_339: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_63, sum_141);  alias_63 = sum_141 = None
    sub_130: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_340: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_130, 0.14433756729740643);  sub_130 = None
    view_451: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_340, [128, 196, 196]);  mul_340 = None
    permute_324: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    bmm_50: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_324, view_451);  permute_324 = None
    permute_325: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_51: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_451, permute_325);  view_451 = permute_325 = None
    view_452: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_50, [8, 16, 48, 196]);  bmm_50 = None
    view_453: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_51, [8, 16, 196, 48]);  bmm_51 = None
    permute_326: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_452, [0, 1, 3, 2]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_327: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_129, [0, 2, 3, 1]);  sub_129 = None
    sum_142: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_327, [0, 1, 2], True)
    view_454: "f32[16]" = torch.ops.aten.view.default(sum_142, [16]);  sum_142 = None
    clone_200: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_455: "f32[307328, 16]" = torch.ops.aten.view.default(clone_200, [307328, 16]);  clone_200 = None
    permute_328: "f32[16, 307328]" = torch.ops.aten.permute.default(view_455, [1, 0]);  view_455 = None
    mm_102: "f32[16, 3]" = torch.ops.aten.mm.default(permute_328, view_138);  permute_328 = view_138 = None
    permute_329: "f32[3, 16]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    permute_330: "f32[16, 3]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_20: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_39: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_20, permute_326, 0, 1);  full_20 = permute_326 = None
    full_21: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_40: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_21, view_453, 0, 0);  full_21 = view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_162: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_39, select_scatter_40);  select_scatter_39 = select_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_331: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_162, [1, 3, 0, 2, 4]);  add_162 = None
    clone_201: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_456: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_201, [8, 196, 1536]);  clone_201 = None
    view_457: "f32[1568, 1536]" = torch.ops.aten.view.default(view_456, [1568, 1536]);  view_456 = None
    permute_332: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_103: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_332, view_135);  permute_332 = view_135 = None
    permute_333: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    permute_334: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_104: "f32[1568, 768]" = torch.ops.aten.mm.default(view_457, permute_334);  view_457 = permute_334 = None
    view_458: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_104, [8, 196, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_163: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_449, view_458);  view_449 = view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_335: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_202: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    sub_131: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_202, getitem_21);  clone_202 = getitem_21 = None
    mul_341: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_10);  sub_131 = None
    mul_342: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_163, primals_28);  primals_28 = None
    mul_343: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
    sum_143: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
    mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_341);  mul_342 = None
    sum_144: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
    mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_341, sum_144);  sum_144 = None
    sub_132: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_143);  mul_343 = sum_143 = None
    sub_133: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_345);  sub_132 = mul_345 = None
    div_61: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_346: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_133);  div_61 = sub_133 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_163, mul_341);  mul_341 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_164: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_159, mul_346);  add_159 = mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[1568, 768]" = torch.ops.aten.view.default(add_164, [1568, 768])
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_105: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_459, permute_336);  permute_336 = None
    permute_337: "f32[768, 1568]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_106: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_337, view_129);  permute_337 = view_129 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_339: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_461: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_105, [8, 196, 3072]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_348: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_19: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_348);  mul_348 = None
    add_165: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_349: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_165, 0.5);  add_165 = None
    mul_350: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_351: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_350, -0.5);  mul_350 = None
    exp_29: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_351);  mul_351 = None
    mul_352: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_353: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, mul_352);  view_128 = mul_352 = None
    add_166: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_349, mul_353);  mul_349 = mul_353 = None
    mul_354: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_461, add_166);  view_461 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_354, [1568, 3072]);  mul_354 = None
    permute_340: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_107: "f32[1568, 768]" = torch.ops.aten.mm.default(view_462, permute_340);  permute_340 = None
    permute_341: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_108: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_341, view_127);  permute_341 = view_127 = None
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_464: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_107, [8, 196, 768]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_203: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format);  add_46 = None
    sub_134: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_203, getitem_19);  clone_203 = getitem_19 = None
    mul_355: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_9);  sub_134 = None
    mul_356: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_464, primals_26);  primals_26 = None
    mul_357: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_356, 768)
    sum_149: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
    mul_358: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_355);  mul_356 = None
    sum_150: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
    mul_359: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_355, sum_150);  sum_150 = None
    sub_135: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_357, sum_149);  mul_357 = sum_149 = None
    sub_136: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_359);  sub_135 = mul_359 = None
    div_62: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_360: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_62, sub_136);  div_62 = sub_136 = None
    mul_361: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_464, mul_355);  mul_355 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_464, [0, 1]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_167: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_164, mul_360);  add_164 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_465: "f32[1568, 768]" = torch.ops.aten.view.default(add_167, [1568, 768])
    permute_344: "f32[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_109: "f32[1568, 768]" = torch.ops.aten.mm.default(view_465, permute_344);  permute_344 = None
    permute_345: "f32[768, 1568]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_110: "f32[768, 768]" = torch.ops.aten.mm.default(permute_345, view_125);  permute_345 = view_125 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_467: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_109, [8, 196, 768]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_468: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_467, [8, 196, 16, 48]);  view_467 = None
    permute_348: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_204: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_469: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_204, [128, 196, 48]);  clone_204 = None
    permute_349: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    bmm_52: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_349, view_469);  permute_349 = None
    permute_350: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_53: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_469, permute_350);  view_469 = permute_350 = None
    view_470: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_52, [8, 16, 196, 48]);  bmm_52 = None
    view_471: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_53, [8, 16, 196, 196]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_351: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    clone_205: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_472: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_205, [8, 196, 768]);  clone_205 = None
    view_473: "f32[1568, 768]" = torch.ops.aten.view.default(view_472, [1568, 768]);  view_472 = None
    permute_352: "f32[768, 1568]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_352, view_118);  permute_352 = view_118 = None
    permute_353: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_112: "f32[1568, 768]" = torch.ops.aten.mm.default(view_473, permute_354);  view_473 = permute_354 = None
    view_474: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_112, [8, 196, 768]);  mm_112 = None
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_63: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_68, unsqueeze_29);  clone_68 = None
    div_64: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_63, unsqueeze_29);  div_63 = None
    neg_10: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_471)
    mul_362: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_10, div_64);  neg_10 = div_64 = None
    div_65: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_471, unsqueeze_29);  view_471 = unsqueeze_29 = None
    sum_154: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [3], True);  mul_362 = None
    squeeze_5: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_154, -1);  sum_154 = None
    unsqueeze_65: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_5, -1);  squeeze_5 = None
    expand_84: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_65, [8, 16, 196, 196]);  unsqueeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_168: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_65, expand_84);  div_65 = expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_363: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, sigmoid_9);  sigmoid_9 = None
    mul_364: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, div_13);  div_13 = None
    sum_155: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3], True);  mul_364 = None
    alias_64: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_137: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_64)
    mul_365: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_64, sub_137);  alias_64 = sub_137 = None
    mul_366: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_155, mul_365);  sum_155 = mul_365 = None
    mul_367: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, sub_28);  sub_28 = None
    mul_368: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, div_12);  add_168 = div_12 = None
    sum_156: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3], True);  mul_368 = None
    neg_11: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_156);  sum_156 = None
    alias_65: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    sub_138: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_65)
    mul_369: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_65, sub_138);  alias_65 = sub_138 = None
    mul_370: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_11, mul_369);  neg_11 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_169: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_366, mul_370);  mul_366 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_475: "f32[16]" = torch.ops.aten.view.default(add_169, [16]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_66: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_371: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_363, alias_66);  mul_363 = None
    sum_157: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [-1], True)
    mul_372: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_66, sum_157);  alias_66 = sum_157 = None
    sub_139: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_67: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_373: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_367, alias_67);  mul_367 = None
    sum_158: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [-1], True)
    mul_374: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_67, sum_158);  alias_67 = sum_158 = None
    sub_140: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_375: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_140, 0.14433756729740643);  sub_140 = None
    view_476: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_375, [128, 196, 196]);  mul_375 = None
    permute_356: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    bmm_54: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_356, view_476);  permute_356 = None
    permute_357: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    bmm_55: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_476, permute_357);  view_476 = permute_357 = None
    view_477: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_54, [8, 16, 48, 196]);  bmm_54 = None
    view_478: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_55, [8, 16, 196, 48]);  bmm_55 = None
    permute_358: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_477, [0, 1, 3, 2]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_359: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_139, [0, 2, 3, 1]);  sub_139 = None
    sum_159: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_359, [0, 1, 2], True)
    view_479: "f32[16]" = torch.ops.aten.view.default(sum_159, [16]);  sum_159 = None
    clone_206: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_480: "f32[307328, 16]" = torch.ops.aten.view.default(clone_206, [307328, 16]);  clone_206 = None
    permute_360: "f32[16, 307328]" = torch.ops.aten.permute.default(view_480, [1, 0]);  view_480 = None
    mm_113: "f32[16, 3]" = torch.ops.aten.mm.default(permute_360, view_112);  permute_360 = view_112 = None
    permute_361: "f32[3, 16]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    permute_362: "f32[16, 3]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_22: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_41: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_22, permute_358, 0, 1);  full_22 = permute_358 = None
    full_23: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_42: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_23, view_478, 0, 0);  full_23 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_170: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_41, select_scatter_42);  select_scatter_41 = select_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_363: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_170, [1, 3, 0, 2, 4]);  add_170 = None
    clone_207: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    view_481: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_207, [8, 196, 1536]);  clone_207 = None
    view_482: "f32[1568, 1536]" = torch.ops.aten.view.default(view_481, [1568, 1536]);  view_481 = None
    permute_364: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_114: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_364, view_109);  permute_364 = view_109 = None
    permute_365: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    permute_366: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_115: "f32[1568, 768]" = torch.ops.aten.mm.default(view_482, permute_366);  view_482 = permute_366 = None
    view_483: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_115, [8, 196, 768]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_171: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_474, view_483);  view_474 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_367: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_208: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_40, memory_format = torch.contiguous_format);  add_40 = None
    sub_141: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_208, getitem_17);  clone_208 = getitem_17 = None
    mul_376: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_8);  sub_141 = None
    mul_377: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_171, primals_23);  primals_23 = None
    mul_378: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_377, 768)
    sum_160: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
    mul_379: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_377, mul_376);  mul_377 = None
    sum_161: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
    mul_380: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_376, sum_161);  sum_161 = None
    sub_142: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_378, sum_160);  mul_378 = sum_160 = None
    sub_143: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_142, mul_380);  sub_142 = mul_380 = None
    div_66: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_381: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_143);  div_66 = sub_143 = None
    mul_382: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_376);  mul_376 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_167, mul_381);  add_167 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_484: "f32[1568, 768]" = torch.ops.aten.view.default(add_172, [1568, 768])
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_116: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_484, permute_368);  permute_368 = None
    permute_369: "f32[768, 1568]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_117: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_103);  permute_369 = view_103 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[768]" = torch.ops.aten.view.default(sum_164, [768]);  sum_164 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_486: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_116, [8, 196, 3072]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_383: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_20: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_383);  mul_383 = None
    add_173: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_384: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_173, 0.5);  add_173 = None
    mul_385: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, view_102)
    mul_386: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_385, -0.5);  mul_385 = None
    exp_30: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_386);  mul_386 = None
    mul_387: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_388: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, mul_387);  view_102 = mul_387 = None
    add_174: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_384, mul_388);  mul_384 = mul_388 = None
    mul_389: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_486, add_174);  view_486 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_487: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_389, [1568, 3072]);  mul_389 = None
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_118: "f32[1568, 768]" = torch.ops.aten.mm.default(view_487, permute_372);  permute_372 = None
    permute_373: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_119: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_101);  permute_373 = view_101 = None
    permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_165: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[3072]" = torch.ops.aten.view.default(sum_165, [3072]);  sum_165 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_489: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_118, [8, 196, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_209: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format);  add_36 = None
    sub_144: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_209, getitem_15);  clone_209 = getitem_15 = None
    mul_390: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_7);  sub_144 = None
    mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_489, primals_21);  primals_21 = None
    mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_391, 768)
    sum_166: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True)
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_390);  mul_391 = None
    sum_167: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True);  mul_393 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_390, sum_167);  sum_167 = None
    sub_145: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_392, sum_166);  mul_392 = sum_166 = None
    sub_146: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_145, mul_394);  sub_145 = mul_394 = None
    div_67: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_395: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_146);  div_67 = sub_146 = None
    mul_396: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_489, mul_390);  mul_390 = None
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 1]);  mul_396 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_489, [0, 1]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_175: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_172, mul_395);  add_172 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_490: "f32[1568, 768]" = torch.ops.aten.view.default(add_175, [1568, 768])
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_120: "f32[1568, 768]" = torch.ops.aten.mm.default(view_490, permute_376);  permute_376 = None
    permute_377: "f32[768, 1568]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_377, view_99);  permute_377 = view_99 = None
    permute_378: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_492: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_120, [8, 196, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_493: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_492, [8, 196, 16, 48]);  view_492 = None
    permute_380: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_210: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
    view_494: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_210, [128, 196, 48]);  clone_210 = None
    permute_381: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    bmm_56: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_381, view_494);  permute_381 = None
    permute_382: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    bmm_57: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_494, permute_382);  view_494 = permute_382 = None
    view_495: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_56, [8, 16, 196, 48]);  bmm_56 = None
    view_496: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_57, [8, 16, 196, 196]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_383: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
    clone_211: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    view_497: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_211, [8, 196, 768]);  clone_211 = None
    view_498: "f32[1568, 768]" = torch.ops.aten.view.default(view_497, [1568, 768]);  view_497 = None
    permute_384: "f32[768, 1568]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_122: "f32[768, 768]" = torch.ops.aten.mm.default(permute_384, view_92);  permute_384 = view_92 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    permute_386: "f32[768, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_123: "f32[1568, 768]" = torch.ops.aten.mm.default(view_498, permute_386);  view_498 = permute_386 = None
    view_499: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_123, [8, 196, 768]);  mm_123 = None
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_68: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_53, unsqueeze_23);  clone_53 = None
    div_69: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_68, unsqueeze_23);  div_68 = None
    neg_12: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_496)
    mul_397: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_12, div_69);  neg_12 = div_69 = None
    div_70: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_496, unsqueeze_23);  view_496 = unsqueeze_23 = None
    sum_171: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [3], True);  mul_397 = None
    squeeze_6: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_171, -1);  sum_171 = None
    unsqueeze_66: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_6, -1);  squeeze_6 = None
    expand_85: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_66, [8, 16, 196, 196]);  unsqueeze_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_176: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_70, expand_85);  div_70 = expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_398: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_7);  sigmoid_7 = None
    mul_399: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, div_10);  div_10 = None
    sum_172: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3], True);  mul_399 = None
    alias_68: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_147: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_68)
    mul_400: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_68, sub_147);  alias_68 = sub_147 = None
    mul_401: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_172, mul_400);  sum_172 = mul_400 = None
    mul_402: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, sub_22);  sub_22 = None
    mul_403: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, div_9);  add_176 = div_9 = None
    sum_173: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 2, 3], True);  mul_403 = None
    neg_13: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_173);  sum_173 = None
    alias_69: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_148: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_69)
    mul_404: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_69, sub_148);  alias_69 = sub_148 = None
    mul_405: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_13, mul_404);  neg_13 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_177: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_500: "f32[16]" = torch.ops.aten.view.default(add_177, [16]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_70: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_406: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_398, alias_70);  mul_398 = None
    sum_174: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [-1], True)
    mul_407: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_70, sum_174);  alias_70 = sum_174 = None
    sub_149: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_71: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_408: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_402, alias_71);  mul_402 = None
    sum_175: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [-1], True)
    mul_409: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_71, sum_175);  alias_71 = sum_175 = None
    sub_150: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_410: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_150, 0.14433756729740643);  sub_150 = None
    view_501: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_410, [128, 196, 196]);  mul_410 = None
    permute_388: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    bmm_58: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_388, view_501);  permute_388 = None
    permute_389: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    bmm_59: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_501, permute_389);  view_501 = permute_389 = None
    view_502: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_58, [8, 16, 48, 196]);  bmm_58 = None
    view_503: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_59, [8, 16, 196, 48]);  bmm_59 = None
    permute_390: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 2]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_391: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_149, [0, 2, 3, 1]);  sub_149 = None
    sum_176: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_391, [0, 1, 2], True)
    view_504: "f32[16]" = torch.ops.aten.view.default(sum_176, [16]);  sum_176 = None
    clone_212: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_505: "f32[307328, 16]" = torch.ops.aten.view.default(clone_212, [307328, 16]);  clone_212 = None
    permute_392: "f32[16, 307328]" = torch.ops.aten.permute.default(view_505, [1, 0]);  view_505 = None
    mm_124: "f32[16, 3]" = torch.ops.aten.mm.default(permute_392, view_86);  permute_392 = view_86 = None
    permute_393: "f32[3, 16]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    permute_394: "f32[16, 3]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_24: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_43: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_24, permute_390, 0, 1);  full_24 = permute_390 = None
    full_25: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_44: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_25, view_503, 0, 0);  full_25 = view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_178: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_43, select_scatter_44);  select_scatter_43 = select_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_395: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_178, [1, 3, 0, 2, 4]);  add_178 = None
    clone_213: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_506: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_213, [8, 196, 1536]);  clone_213 = None
    view_507: "f32[1568, 1536]" = torch.ops.aten.view.default(view_506, [1568, 1536]);  view_506 = None
    permute_396: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_125: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_396, view_83);  permute_396 = view_83 = None
    permute_397: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    permute_398: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_126: "f32[1568, 768]" = torch.ops.aten.mm.default(view_507, permute_398);  view_507 = permute_398 = None
    view_508: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_126, [8, 196, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_179: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_499, view_508);  view_499 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_399: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_214: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format);  add_30 = None
    sub_151: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_214, getitem_13);  clone_214 = getitem_13 = None
    mul_411: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_6);  sub_151 = None
    mul_412: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_179, primals_18);  primals_18 = None
    mul_413: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
    sum_177: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_411);  mul_412 = None
    sum_178: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_411, sum_178);  sum_178 = None
    sub_152: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_177);  mul_413 = sum_177 = None
    sub_153: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_152, mul_415);  sub_152 = mul_415 = None
    div_71: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_416: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_71, sub_153);  div_71 = sub_153 = None
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_411);  mul_411 = None
    sum_179: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_180: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_175, mul_416);  add_175 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_509: "f32[1568, 768]" = torch.ops.aten.view.default(add_180, [1568, 768])
    permute_400: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_127: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_509, permute_400);  permute_400 = None
    permute_401: "f32[768, 1568]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_128: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_401, view_77);  permute_401 = view_77 = None
    permute_402: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[768]" = torch.ops.aten.view.default(sum_181, [768]);  sum_181 = None
    permute_403: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_511: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_127, [8, 196, 3072]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_418: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_21: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_418);  mul_418 = None
    add_181: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_419: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
    mul_420: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, view_76)
    mul_421: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_420, -0.5);  mul_420 = None
    exp_31: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_421);  mul_421 = None
    mul_422: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_423: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, mul_422);  view_76 = mul_422 = None
    add_182: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_419, mul_423);  mul_419 = mul_423 = None
    mul_424: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_511, add_182);  view_511 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_512: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_424, [1568, 3072]);  mul_424 = None
    permute_404: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_129: "f32[1568, 768]" = torch.ops.aten.mm.default(view_512, permute_404);  permute_404 = None
    permute_405: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_130: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_405, view_75);  permute_405 = view_75 = None
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_182: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True);  view_512 = None
    view_513: "f32[3072]" = torch.ops.aten.view.default(sum_182, [3072]);  sum_182 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_514: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_129, [8, 196, 768]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_215: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format);  add_26 = None
    sub_154: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_215, getitem_11);  clone_215 = getitem_11 = None
    mul_425: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_5);  sub_154 = None
    mul_426: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_514, primals_16);  primals_16 = None
    mul_427: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_426, 768)
    sum_183: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [2], True)
    mul_428: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_426, mul_425);  mul_426 = None
    sum_184: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True);  mul_428 = None
    mul_429: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_425, sum_184);  sum_184 = None
    sub_155: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_427, sum_183);  mul_427 = sum_183 = None
    sub_156: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_429);  sub_155 = mul_429 = None
    div_72: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_430: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_156);  div_72 = sub_156 = None
    mul_431: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_514, mul_425);  mul_425 = None
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1]);  mul_431 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_514, [0, 1]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_183: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_180, mul_430);  add_180 = mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_515: "f32[1568, 768]" = torch.ops.aten.view.default(add_183, [1568, 768])
    permute_408: "f32[768, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_131: "f32[1568, 768]" = torch.ops.aten.mm.default(view_515, permute_408);  permute_408 = None
    permute_409: "f32[768, 1568]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_132: "f32[768, 768]" = torch.ops.aten.mm.default(permute_409, view_73);  permute_409 = view_73 = None
    permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_187, [768]);  sum_187 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_517: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_131, [8, 196, 768]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_518: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_517, [8, 196, 16, 48]);  view_517 = None
    permute_412: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
    clone_216: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_519: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_216, [128, 196, 48]);  clone_216 = None
    permute_413: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    bmm_60: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_413, view_519);  permute_413 = None
    permute_414: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    bmm_61: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_519, permute_414);  view_519 = permute_414 = None
    view_520: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_60, [8, 16, 196, 48]);  bmm_60 = None
    view_521: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_61, [8, 16, 196, 196]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_415: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    clone_217: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    view_522: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_217, [8, 196, 768]);  clone_217 = None
    view_523: "f32[1568, 768]" = torch.ops.aten.view.default(view_522, [1568, 768]);  view_522 = None
    permute_416: "f32[768, 1568]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_66);  permute_416 = view_66 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_134: "f32[1568, 768]" = torch.ops.aten.mm.default(view_523, permute_418);  view_523 = permute_418 = None
    view_524: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_134, [8, 196, 768]);  mm_134 = None
    permute_419: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_73: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_38, unsqueeze_17);  clone_38 = None
    div_74: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_73, unsqueeze_17);  div_73 = None
    neg_14: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_521)
    mul_432: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_14, div_74);  neg_14 = div_74 = None
    div_75: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_521, unsqueeze_17);  view_521 = unsqueeze_17 = None
    sum_188: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [3], True);  mul_432 = None
    squeeze_7: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_188, -1);  sum_188 = None
    unsqueeze_67: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_7, -1);  squeeze_7 = None
    expand_86: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_67, [8, 16, 196, 196]);  unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_184: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_75, expand_86);  div_75 = expand_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_433: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, sigmoid_5);  sigmoid_5 = None
    mul_434: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, div_7);  div_7 = None
    sum_189: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2, 3], True);  mul_434 = None
    alias_72: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_157: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_72)
    mul_435: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_72, sub_157);  alias_72 = sub_157 = None
    mul_436: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_189, mul_435);  sum_189 = mul_435 = None
    mul_437: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, sub_16);  sub_16 = None
    mul_438: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, div_6);  add_184 = div_6 = None
    sum_190: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3], True);  mul_438 = None
    neg_15: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_190);  sum_190 = None
    alias_73: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_158: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_73)
    mul_439: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_73, sub_158);  alias_73 = sub_158 = None
    mul_440: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_15, mul_439);  neg_15 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_185: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_525: "f32[16]" = torch.ops.aten.view.default(add_185, [16]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_74: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_441: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_433, alias_74);  mul_433 = None
    sum_191: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [-1], True)
    mul_442: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_74, sum_191);  alias_74 = sum_191 = None
    sub_159: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_75: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_443: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_437, alias_75);  mul_437 = None
    sum_192: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [-1], True)
    mul_444: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_75, sum_192);  alias_75 = sum_192 = None
    sub_160: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_445: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_160, 0.14433756729740643);  sub_160 = None
    view_526: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_445, [128, 196, 196]);  mul_445 = None
    permute_420: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_62: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_420, view_526);  permute_420 = None
    permute_421: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_63: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_526, permute_421);  view_526 = permute_421 = None
    view_527: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_62, [8, 16, 48, 196]);  bmm_62 = None
    view_528: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_63, [8, 16, 196, 48]);  bmm_63 = None
    permute_422: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_527, [0, 1, 3, 2]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_423: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_159, [0, 2, 3, 1]);  sub_159 = None
    sum_193: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_423, [0, 1, 2], True)
    view_529: "f32[16]" = torch.ops.aten.view.default(sum_193, [16]);  sum_193 = None
    clone_218: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
    view_530: "f32[307328, 16]" = torch.ops.aten.view.default(clone_218, [307328, 16]);  clone_218 = None
    permute_424: "f32[16, 307328]" = torch.ops.aten.permute.default(view_530, [1, 0]);  view_530 = None
    mm_135: "f32[16, 3]" = torch.ops.aten.mm.default(permute_424, view_60);  permute_424 = view_60 = None
    permute_425: "f32[3, 16]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    permute_426: "f32[16, 3]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_26: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_45: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_26, permute_422, 0, 1);  full_26 = permute_422 = None
    full_27: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_27, view_528, 0, 0);  full_27 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_186: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_45, select_scatter_46);  select_scatter_45 = select_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_427: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_186, [1, 3, 0, 2, 4]);  add_186 = None
    clone_219: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    view_531: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_219, [8, 196, 1536]);  clone_219 = None
    view_532: "f32[1568, 1536]" = torch.ops.aten.view.default(view_531, [1568, 1536]);  view_531 = None
    permute_428: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_136: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_428, view_57);  permute_428 = view_57 = None
    permute_429: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    permute_430: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_137: "f32[1568, 768]" = torch.ops.aten.mm.default(view_532, permute_430);  view_532 = permute_430 = None
    view_533: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_137, [8, 196, 768]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_187: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_524, view_533);  view_524 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_431: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_220: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format);  add_20 = None
    sub_161: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_220, getitem_9);  clone_220 = getitem_9 = None
    mul_446: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_4);  sub_161 = None
    mul_447: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_187, primals_13);  primals_13 = None
    mul_448: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_447, 768)
    sum_194: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
    mul_449: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_447, mul_446);  mul_447 = None
    sum_195: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
    mul_450: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_446, sum_195);  sum_195 = None
    sub_162: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_448, sum_194);  mul_448 = sum_194 = None
    sub_163: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_162, mul_450);  sub_162 = mul_450 = None
    div_76: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_451: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_76, sub_163);  div_76 = sub_163 = None
    mul_452: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_187, mul_446);  mul_446 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_187, [0, 1]);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_188: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_183, mul_451);  add_183 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_534: "f32[1568, 768]" = torch.ops.aten.view.default(add_188, [1568, 768])
    permute_432: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_138: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_534, permute_432);  permute_432 = None
    permute_433: "f32[768, 1568]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_139: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_433, view_51);  permute_433 = view_51 = None
    permute_434: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    permute_435: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_536: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_138, [8, 196, 3072]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_453: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_22: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_453);  mul_453 = None
    add_189: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_454: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_189, 0.5);  add_189 = None
    mul_455: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, view_50)
    mul_456: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_455, -0.5);  mul_455 = None
    exp_32: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_456);  mul_456 = None
    mul_457: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_458: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, mul_457);  view_50 = mul_457 = None
    add_190: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_454, mul_458);  mul_454 = mul_458 = None
    mul_459: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_536, add_190);  view_536 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_537: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_459, [1568, 3072]);  mul_459 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_140: "f32[1568, 768]" = torch.ops.aten.mm.default(view_537, permute_436);  permute_436 = None
    permute_437: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_141: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_437, view_49);  permute_437 = view_49 = None
    permute_438: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[3072]" = torch.ops.aten.view.default(sum_199, [3072]);  sum_199 = None
    permute_439: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_539: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_140, [8, 196, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_221: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format);  add_16 = None
    sub_164: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_221, getitem_7);  clone_221 = getitem_7 = None
    mul_460: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_3);  sub_164 = None
    mul_461: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_539, primals_11);  primals_11 = None
    mul_462: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_461, 768)
    sum_200: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
    mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_461, mul_460);  mul_461 = None
    sum_201: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
    mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_460, sum_201);  sum_201 = None
    sub_165: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_462, sum_200);  mul_462 = sum_200 = None
    sub_166: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_165, mul_464);  sub_165 = mul_464 = None
    div_77: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_465: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_77, sub_166);  div_77 = sub_166 = None
    mul_466: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_539, mul_460);  mul_460 = None
    sum_202: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
    sum_203: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_539, [0, 1]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_191: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_188, mul_465);  add_188 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_540: "f32[1568, 768]" = torch.ops.aten.view.default(add_191, [1568, 768])
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_142: "f32[1568, 768]" = torch.ops.aten.mm.default(view_540, permute_440);  permute_440 = None
    permute_441: "f32[768, 1568]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_441, view_47);  permute_441 = view_47 = None
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_204: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[768]" = torch.ops.aten.view.default(sum_204, [768]);  sum_204 = None
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_542: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_142, [8, 196, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_543: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_542, [8, 196, 16, 48]);  view_542 = None
    permute_444: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    clone_222: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_444, memory_format = torch.contiguous_format);  permute_444 = None
    view_544: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_222, [128, 196, 48]);  clone_222 = None
    permute_445: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    bmm_64: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_445, view_544);  permute_445 = None
    permute_446: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_65: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_544, permute_446);  view_544 = permute_446 = None
    view_545: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_64, [8, 16, 196, 48]);  bmm_64 = None
    view_546: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_65, [8, 16, 196, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_447: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
    clone_223: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_547: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_223, [8, 196, 768]);  clone_223 = None
    view_548: "f32[1568, 768]" = torch.ops.aten.view.default(view_547, [1568, 768]);  view_547 = None
    permute_448: "f32[768, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_40);  permute_448 = view_40 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_145: "f32[1568, 768]" = torch.ops.aten.mm.default(view_548, permute_450);  view_548 = permute_450 = None
    view_549: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_145, [8, 196, 768]);  mm_145 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_78: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_23, unsqueeze_11);  clone_23 = None
    div_79: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_78, unsqueeze_11);  div_78 = None
    neg_16: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_546)
    mul_467: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_16, div_79);  neg_16 = div_79 = None
    div_80: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_546, unsqueeze_11);  view_546 = unsqueeze_11 = None
    sum_205: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [3], True);  mul_467 = None
    squeeze_8: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_205, -1);  sum_205 = None
    unsqueeze_68: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_8, -1);  squeeze_8 = None
    expand_87: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_68, [8, 16, 196, 196]);  unsqueeze_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_192: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_80, expand_87);  div_80 = expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_468: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, sigmoid_3);  sigmoid_3 = None
    mul_469: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, div_4);  div_4 = None
    sum_206: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3], True);  mul_469 = None
    alias_76: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_167: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_76)
    mul_470: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_76, sub_167);  alias_76 = sub_167 = None
    mul_471: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_206, mul_470);  sum_206 = mul_470 = None
    mul_472: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, sub_10);  sub_10 = None
    mul_473: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, div_3);  add_192 = div_3 = None
    sum_207: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3], True);  mul_473 = None
    neg_17: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_207);  sum_207 = None
    alias_77: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    sub_168: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_77)
    mul_474: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_77, sub_168);  alias_77 = sub_168 = None
    mul_475: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_17, mul_474);  neg_17 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_193: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_471, mul_475);  mul_471 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_550: "f32[16]" = torch.ops.aten.view.default(add_193, [16]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_78: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_476: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_468, alias_78);  mul_468 = None
    sum_208: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
    mul_477: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_78, sum_208);  alias_78 = sum_208 = None
    sub_169: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_79: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_478: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_472, alias_79);  mul_472 = None
    sum_209: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
    mul_479: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_79, sum_209);  alias_79 = sum_209 = None
    sub_170: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_480: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_170, 0.14433756729740643);  sub_170 = None
    view_551: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_480, [128, 196, 196]);  mul_480 = None
    permute_452: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_66: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_452, view_551);  permute_452 = None
    permute_453: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_67: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_551, permute_453);  view_551 = permute_453 = None
    view_552: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_66, [8, 16, 48, 196]);  bmm_66 = None
    view_553: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_67, [8, 16, 196, 48]);  bmm_67 = None
    permute_454: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_552, [0, 1, 3, 2]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_455: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_169, [0, 2, 3, 1]);  sub_169 = None
    sum_210: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_455, [0, 1, 2], True)
    view_554: "f32[16]" = torch.ops.aten.view.default(sum_210, [16]);  sum_210 = None
    clone_224: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    view_555: "f32[307328, 16]" = torch.ops.aten.view.default(clone_224, [307328, 16]);  clone_224 = None
    permute_456: "f32[16, 307328]" = torch.ops.aten.permute.default(view_555, [1, 0]);  view_555 = None
    mm_146: "f32[16, 3]" = torch.ops.aten.mm.default(permute_456, view_34);  permute_456 = view_34 = None
    permute_457: "f32[3, 16]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    permute_458: "f32[16, 3]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_28: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_47: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_28, permute_454, 0, 1);  full_28 = permute_454 = None
    full_29: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_48: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_29, view_553, 0, 0);  full_29 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_194: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_47, select_scatter_48);  select_scatter_47 = select_scatter_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_459: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_194, [1, 3, 0, 2, 4]);  add_194 = None
    clone_225: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_556: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_225, [8, 196, 1536]);  clone_225 = None
    view_557: "f32[1568, 1536]" = torch.ops.aten.view.default(view_556, [1568, 1536]);  view_556 = None
    permute_460: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_147: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_460, view_31);  permute_460 = view_31 = None
    permute_461: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    permute_462: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_148: "f32[1568, 768]" = torch.ops.aten.mm.default(view_557, permute_462);  view_557 = permute_462 = None
    view_558: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_148, [8, 196, 768]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_195: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_549, view_558);  view_549 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_463: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_226: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format);  add_10 = None
    sub_171: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_226, getitem_5);  clone_226 = getitem_5 = None
    mul_481: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_171, rsqrt_2);  sub_171 = None
    mul_482: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_195, primals_8);  primals_8 = None
    mul_483: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, 768)
    sum_211: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, mul_481);  mul_482 = None
    sum_212: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_481, sum_212);  sum_212 = None
    sub_172: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_483, sum_211);  mul_483 = sum_211 = None
    sub_173: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_172, mul_485);  sub_172 = mul_485 = None
    div_81: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_486: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_81, sub_173);  div_81 = sub_173 = None
    mul_487: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_195, mul_481);  mul_481 = None
    sum_213: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_196: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_191, mul_486);  add_191 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_559: "f32[1568, 768]" = torch.ops.aten.view.default(add_196, [1568, 768])
    permute_464: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_149: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_559, permute_464);  permute_464 = None
    permute_465: "f32[768, 1568]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_150: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_465, view_25);  permute_465 = view_25 = None
    permute_466: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_215: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[768]" = torch.ops.aten.view.default(sum_215, [768]);  sum_215 = None
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_561: "f32[8, 196, 3072]" = torch.ops.aten.view.default(mm_149, [8, 196, 3072]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_488: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476)
    erf_23: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_488);  mul_488 = None
    add_197: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_489: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_197, 0.5);  add_197 = None
    mul_490: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, view_24)
    mul_491: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
    exp_33: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
    mul_492: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_493: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, mul_492);  view_24 = mul_492 = None
    add_198: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
    mul_494: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_561, add_198);  view_561 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_562: "f32[1568, 3072]" = torch.ops.aten.view.default(mul_494, [1568, 3072]);  mul_494 = None
    permute_468: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_151: "f32[1568, 768]" = torch.ops.aten.mm.default(view_562, permute_468);  permute_468 = None
    permute_469: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_152: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_469, view_23);  permute_469 = view_23 = None
    permute_470: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_216: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[3072]" = torch.ops.aten.view.default(sum_216, [3072]);  sum_216 = None
    permute_471: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_564: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_151, [8, 196, 768]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_227: "f32[8, 196, 768]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format);  add_6 = None
    sub_174: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_227, getitem_3);  clone_227 = getitem_3 = None
    mul_495: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_174, rsqrt_1);  sub_174 = None
    mul_496: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_564, primals_6);  primals_6 = None
    mul_497: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_496, 768)
    sum_217: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
    mul_498: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_496, mul_495);  mul_496 = None
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
    mul_499: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_495, sum_218);  sum_218 = None
    sub_175: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_497, sum_217);  mul_497 = sum_217 = None
    sub_176: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_175, mul_499);  sub_175 = mul_499 = None
    div_82: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_500: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_82, sub_176);  div_82 = sub_176 = None
    mul_501: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_564, mul_495);  mul_495 = None
    sum_219: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_564, [0, 1]);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_199: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_196, mul_500);  add_196 = mul_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_565: "f32[1568, 768]" = torch.ops.aten.view.default(add_199, [1568, 768])
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_153: "f32[1568, 768]" = torch.ops.aten.mm.default(view_565, permute_472);  permute_472 = None
    permute_473: "f32[768, 1568]" = torch.ops.aten.permute.default(view_565, [1, 0])
    mm_154: "f32[768, 768]" = torch.ops.aten.mm.default(permute_473, view_21);  permute_473 = view_21 = None
    permute_474: "f32[768, 768]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_221: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
    view_566: "f32[768]" = torch.ops.aten.view.default(sum_221, [768]);  sum_221 = None
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_567: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_153, [8, 196, 768]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_568: "f32[8, 196, 16, 48]" = torch.ops.aten.view.default(view_567, [8, 196, 16, 48]);  view_567 = None
    permute_476: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    clone_228: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_476, memory_format = torch.contiguous_format);  permute_476 = None
    view_569: "f32[128, 196, 48]" = torch.ops.aten.view.default(clone_228, [128, 196, 48]);  clone_228 = None
    permute_477: "f32[128, 196, 196]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    bmm_68: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_477, view_569);  permute_477 = None
    permute_478: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    bmm_69: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_569, permute_478);  view_569 = permute_478 = None
    view_570: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_68, [8, 16, 196, 48]);  bmm_68 = None
    view_571: "f32[8, 16, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 16, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_479: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    clone_229: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_572: "f32[8, 196, 768]" = torch.ops.aten.view.default(clone_229, [8, 196, 768]);  clone_229 = None
    view_573: "f32[1568, 768]" = torch.ops.aten.view.default(view_572, [1568, 768]);  view_572 = None
    permute_480: "f32[768, 1568]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_155: "f32[768, 768]" = torch.ops.aten.mm.default(permute_480, view_14);  permute_480 = view_14 = None
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_156: "f32[1568, 768]" = torch.ops.aten.mm.default(view_573, permute_482);  view_573 = permute_482 = None
    view_574: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_156, [8, 196, 768]);  mm_156 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_83: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(clone_8, unsqueeze_5);  clone_8 = None
    div_84: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_83, unsqueeze_5);  div_83 = None
    neg_18: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_571)
    mul_502: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_18, div_84);  neg_18 = div_84 = None
    div_85: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_571, unsqueeze_5);  view_571 = unsqueeze_5 = None
    sum_222: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [3], True);  mul_502 = None
    squeeze_9: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_222, -1);  sum_222 = None
    unsqueeze_69: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_9, -1);  squeeze_9 = None
    expand_88: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_69, [8, 16, 196, 196]);  unsqueeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_200: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_85, expand_88);  div_85 = expand_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_503: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_1);  sigmoid_1 = None
    mul_504: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, div_1);  div_1 = None
    sum_223: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 2, 3], True);  mul_504 = None
    alias_80: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_177: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_80)
    mul_505: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_80, sub_177);  alias_80 = sub_177 = None
    mul_506: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_223, mul_505);  sum_223 = mul_505 = None
    mul_507: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, sub_4);  sub_4 = None
    mul_508: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, div);  add_200 = div = None
    sum_224: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3], True);  mul_508 = None
    neg_19: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_224);  sum_224 = None
    alias_81: "f32[1, 16, 1, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    sub_178: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_81)
    mul_509: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(alias_81, sub_178);  alias_81 = sub_178 = None
    mul_510: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_19, mul_509);  neg_19 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_201: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_506, mul_510);  mul_506 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_575: "f32[16]" = torch.ops.aten.view.default(add_201, [16]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    alias_82: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_511: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_503, alias_82);  mul_503 = None
    sum_225: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [-1], True)
    mul_512: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_82, sum_225);  alias_82 = sum_225 = None
    sub_179: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    alias_83: "f32[8, 16, 196, 196]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_513: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_507, alias_83);  mul_507 = None
    sum_226: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [-1], True)
    mul_514: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_83, sum_226);  alias_83 = sum_226 = None
    sub_180: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_515: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_180, 0.14433756729740643);  sub_180 = None
    view_576: "f32[128, 196, 196]" = torch.ops.aten.view.default(mul_515, [128, 196, 196]);  mul_515 = None
    permute_484: "f32[128, 48, 196]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_70: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_484, view_576);  permute_484 = None
    permute_485: "f32[128, 196, 48]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_71: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_576, permute_485);  view_576 = permute_485 = None
    view_577: "f32[8, 16, 48, 196]" = torch.ops.aten.view.default(bmm_70, [8, 16, 48, 196]);  bmm_70 = None
    view_578: "f32[8, 16, 196, 48]" = torch.ops.aten.view.default(bmm_71, [8, 16, 196, 48]);  bmm_71 = None
    permute_486: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_577, [0, 1, 3, 2]);  view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_487: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_179, [0, 2, 3, 1]);  sub_179 = None
    sum_227: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_487, [0, 1, 2], True)
    view_579: "f32[16]" = torch.ops.aten.view.default(sum_227, [16]);  sum_227 = None
    clone_230: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_580: "f32[307328, 16]" = torch.ops.aten.view.default(clone_230, [307328, 16]);  clone_230 = None
    permute_488: "f32[16, 307328]" = torch.ops.aten.permute.default(view_580, [1, 0]);  view_580 = None
    mm_157: "f32[16, 3]" = torch.ops.aten.mm.default(permute_488, view_8);  permute_488 = view_8 = None
    permute_489: "f32[3, 16]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    permute_490: "f32[16, 3]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_30: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_49: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_30, permute_486, 0, 1);  full_30 = permute_486 = None
    full_31: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_50: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_31, view_578, 0, 0);  full_31 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_202: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_49, select_scatter_50);  select_scatter_49 = select_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_491: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_202, [1, 3, 0, 2, 4]);  add_202 = None
    clone_231: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_581: "f32[8, 196, 1536]" = torch.ops.aten.view.default(clone_231, [8, 196, 1536]);  clone_231 = None
    view_582: "f32[1568, 1536]" = torch.ops.aten.view.default(view_581, [1568, 1536]);  view_581 = None
    permute_492: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_158: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_492, view_5);  permute_492 = view_5 = None
    permute_493: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    permute_494: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_159: "f32[1568, 768]" = torch.ops.aten.mm.default(view_582, permute_494);  view_582 = permute_494 = None
    view_583: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_159, [8, 196, 768]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_203: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_574, view_583);  view_574 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_495: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_232: "f32[8, 196, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format);  clone = None
    sub_181: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_232, getitem_1);  clone_232 = getitem_1 = None
    mul_516: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt);  sub_181 = None
    mul_517: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_203, primals_3);  primals_3 = None
    mul_518: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_517, 768)
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True)
    mul_519: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_517, mul_516);  mul_517 = None
    sum_229: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True);  mul_519 = None
    mul_520: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_516, sum_229);  sum_229 = None
    sub_182: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_518, sum_228);  mul_518 = sum_228 = None
    sub_183: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_182, mul_520);  sub_182 = mul_520 = None
    div_86: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_521: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_86, sub_183);  div_86 = sub_183 = None
    mul_522: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_203, mul_516);  mul_516 = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 1]);  mul_522 = None
    sum_231: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_204: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_199, mul_521);  add_199 = mul_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    sum_232: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_332, [0], True);  slice_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    sum_233: "f32[1, 196, 768]" = torch.ops.aten.sum.dim_IntList(add_204, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_496: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_204, [0, 2, 1]);  add_204 = None
    view_584: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_496, [8, 768, 14, 14]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_234: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_584, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_584, primals_181, primals_63, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_584 = primals_181 = primals_63 = None
    getitem_57: "f32[768, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return pytree.tree_unflatten([addmm_36, convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, sum_233, sum_232, sum_230, sum_231, view_575, sum_219, sum_220, sum_213, sum_214, view_550, sum_202, sum_203, sum_196, sum_197, view_525, sum_185, sum_186, sum_179, sum_180, view_500, sum_168, sum_169, sum_162, sum_163, view_475, sum_151, sum_152, sum_145, sum_146, view_450, sum_134, sum_135, sum_128, sum_129, view_425, sum_117, sum_118, sum_111, sum_112, view_400, sum_100, sum_101, sum_94, sum_95, view_375, sum_83, sum_84, sum_77, sum_78, view_350, sum_66, sum_67, sum_60, sum_61, sum_54, sum_55, sum_48, sum_49, sum_42, sum_43, sum_36, sum_37, getitem_57, sum_234, permute_495, permute_490, view_579, permute_483, permute_475, view_566, permute_471, view_563, permute_467, view_560, permute_463, permute_458, view_554, permute_451, permute_443, view_541, permute_439, view_538, permute_435, view_535, permute_431, permute_426, view_529, permute_419, permute_411, view_516, permute_407, view_513, permute_403, view_510, permute_399, permute_394, view_504, permute_387, permute_379, view_491, permute_375, view_488, permute_371, view_485, permute_367, permute_362, view_479, permute_355, permute_347, view_466, permute_343, view_463, permute_339, view_460, permute_335, permute_330, view_454, permute_323, permute_315, view_441, permute_311, view_438, permute_307, view_435, permute_303, permute_298, view_429, permute_291, permute_283, view_416, permute_279, view_413, permute_275, view_410, permute_271, permute_266, view_404, permute_259, permute_251, view_391, permute_247, view_388, permute_243, view_385, permute_239, permute_234, view_379, permute_227, permute_219, view_366, permute_215, view_363, permute_211, view_360, permute_207, permute_202, view_354, permute_195, permute_187, view_341, permute_183, view_338, permute_179, view_335, permute_175, permute_164, view_321, permute_160, view_318, permute_156, view_315, permute_152, permute_141, view_301, permute_137, view_298, permute_133, view_295, permute_129, view_293, None], self._out_spec)
    