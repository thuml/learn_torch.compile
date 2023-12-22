from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[8]"; primals_2: "f32[8]"; primals_3: "f32[8]"; primals_4: "f32[8]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[16]"; primals_8: "f32[16]"; primals_9: "f32[32]"; primals_10: "f32[32]"; primals_11: "f32[32]"; primals_12: "f32[32]"; primals_13: "f32[32]"; primals_14: "f32[32]"; primals_15: "f32[32]"; primals_16: "f32[32]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[64]"; primals_20: "f32[64]"; primals_21: "f32[64]"; primals_22: "f32[64]"; primals_23: "f32[64]"; primals_24: "f32[64]"; primals_25: "f32[128]"; primals_26: "f32[128]"; primals_27: "f32[128]"; primals_28: "f32[128]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128]"; primals_32: "f32[128]"; primals_33: "f32[128]"; primals_34: "f32[128]"; primals_35: "f32[128]"; primals_36: "f32[128]"; primals_37: "f32[128]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[128]"; primals_41: "f32[128]"; primals_42: "f32[128]"; primals_43: "f32[128]"; primals_44: "f32[128]"; primals_45: "f32[128]"; primals_46: "f32[128]"; primals_47: "f32[128]"; primals_48: "f32[128]"; primals_49: "f32[256]"; primals_50: "f32[256]"; primals_51: "f32[256]"; primals_52: "f32[256]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[1000, 1280]"; primals_56: "f32[1000]"; primals_57: "f32[8, 3, 3, 3]"; primals_58: "f32[8, 1, 3, 3]"; primals_59: "f32[16, 8, 1, 1]"; primals_60: "f32[16, 1, 3, 3]"; primals_61: "f32[32, 16, 1, 1]"; primals_62: "f32[32, 1, 3, 3]"; primals_63: "f32[32, 32, 1, 1]"; primals_64: "f32[32, 1, 3, 3]"; primals_65: "f32[64, 32, 1, 1]"; primals_66: "f32[64, 1, 3, 3]"; primals_67: "f32[64, 64, 1, 1]"; primals_68: "f32[64, 1, 3, 3]"; primals_69: "f32[128, 64, 1, 1]"; primals_70: "f32[128, 1, 5, 5]"; primals_71: "f32[128, 128, 1, 1]"; primals_72: "f32[128, 1, 5, 5]"; primals_73: "f32[128, 128, 1, 1]"; primals_74: "f32[128, 1, 5, 5]"; primals_75: "f32[128, 128, 1, 1]"; primals_76: "f32[128, 1, 5, 5]"; primals_77: "f32[128, 128, 1, 1]"; primals_78: "f32[128, 1, 5, 5]"; primals_79: "f32[128, 128, 1, 1]"; primals_80: "f32[128, 1, 5, 5]"; primals_81: "f32[32, 128, 1, 1]"; primals_82: "f32[32]"; primals_83: "f32[128, 32, 1, 1]"; primals_84: "f32[128]"; primals_85: "f32[256, 128, 1, 1]"; primals_86: "f32[256, 1, 5, 5]"; primals_87: "f32[64, 256, 1, 1]"; primals_88: "f32[64]"; primals_89: "f32[256, 64, 1, 1]"; primals_90: "f32[256]"; primals_91: "f32[256, 256, 1, 1]"; primals_92: "f32[1280, 256, 1, 1]"; primals_93: "f32[1280]"; primals_94: "i64[]"; primals_95: "f32[8]"; primals_96: "f32[8]"; primals_97: "i64[]"; primals_98: "f32[8]"; primals_99: "f32[8]"; primals_100: "i64[]"; primals_101: "f32[16]"; primals_102: "f32[16]"; primals_103: "i64[]"; primals_104: "f32[16]"; primals_105: "f32[16]"; primals_106: "i64[]"; primals_107: "f32[32]"; primals_108: "f32[32]"; primals_109: "i64[]"; primals_110: "f32[32]"; primals_111: "f32[32]"; primals_112: "i64[]"; primals_113: "f32[32]"; primals_114: "f32[32]"; primals_115: "i64[]"; primals_116: "f32[32]"; primals_117: "f32[32]"; primals_118: "i64[]"; primals_119: "f32[64]"; primals_120: "f32[64]"; primals_121: "i64[]"; primals_122: "f32[64]"; primals_123: "f32[64]"; primals_124: "i64[]"; primals_125: "f32[64]"; primals_126: "f32[64]"; primals_127: "i64[]"; primals_128: "f32[64]"; primals_129: "f32[64]"; primals_130: "i64[]"; primals_131: "f32[128]"; primals_132: "f32[128]"; primals_133: "i64[]"; primals_134: "f32[128]"; primals_135: "f32[128]"; primals_136: "i64[]"; primals_137: "f32[128]"; primals_138: "f32[128]"; primals_139: "i64[]"; primals_140: "f32[128]"; primals_141: "f32[128]"; primals_142: "i64[]"; primals_143: "f32[128]"; primals_144: "f32[128]"; primals_145: "i64[]"; primals_146: "f32[128]"; primals_147: "f32[128]"; primals_148: "i64[]"; primals_149: "f32[128]"; primals_150: "f32[128]"; primals_151: "i64[]"; primals_152: "f32[128]"; primals_153: "f32[128]"; primals_154: "i64[]"; primals_155: "f32[128]"; primals_156: "f32[128]"; primals_157: "i64[]"; primals_158: "f32[128]"; primals_159: "f32[128]"; primals_160: "i64[]"; primals_161: "f32[128]"; primals_162: "f32[128]"; primals_163: "i64[]"; primals_164: "f32[128]"; primals_165: "f32[128]"; primals_166: "i64[]"; primals_167: "f32[256]"; primals_168: "f32[256]"; primals_169: "i64[]"; primals_170: "f32[256]"; primals_171: "f32[256]"; primals_172: "i64[]"; primals_173: "f32[256]"; primals_174: "f32[256]"; primals_175: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(primals_175, primals_57, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_94, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 8, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 8, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[8]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
    add_2: "f32[8]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[8]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[8]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
    add_3: "f32[8]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 8, 112, 112]" = torch.ops.aten.clone.default(add_4)
    add_5: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_7: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  add_4 = clamp_max = None
    div: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(div, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_97, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 8, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 8, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[8]" = torch.ops.aten.mul.Tensor(primals_98, 0.9)
    add_8: "f32[8]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_12: "f32[8]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[8]" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
    add_9: "f32[8]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 8, 112, 112]" = torch.ops.aten.clone.default(add_10)
    add_11: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_1: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_1: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_15: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_1);  add_10 = clamp_max_1 = None
    div_1: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_15, 6);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_1, primals_59, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_100, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_13: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
    add_14: "f32[16]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[16]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
    add_15: "f32[16]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_16: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(add_16)
    add_17: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_16, 3)
    clamp_min_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_17, 0);  add_17 = None
    clamp_max_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_23: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_16, clamp_max_2);  add_16 = clamp_max_2 = None
    div_2: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_23, 6);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(div_2, primals_60, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_103, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 16, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 16, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_19: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_3: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[16]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
    add_20: "f32[16]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_28: "f32[16]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[16]" = torch.ops.aten.mul.Tensor(primals_105, 0.9)
    add_21: "f32[16]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_22: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 16, 56, 56]" = torch.ops.aten.clone.default(add_22)
    add_23: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(add_22, 3)
    clamp_min_3: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_min.default(add_23, 0);  add_23 = None
    clamp_max_3: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_31: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(add_22, clamp_max_3);  add_22 = clamp_max_3 = None
    div_3: "f32[8, 16, 56, 56]" = torch.ops.aten.div.Tensor(mul_31, 6);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_3, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_24: "i64[]" = torch.ops.aten.add.Tensor(primals_106, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 32, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 32, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_25: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_4: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_32: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[32]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
    add_26: "f32[32]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_35: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_36: "f32[32]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[32]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
    add_27: "f32[32]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    unsqueeze_16: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_38: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_17);  mul_32 = unsqueeze_17 = None
    unsqueeze_18: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_28: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_19);  mul_38 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 32, 56, 56]" = torch.ops.aten.clone.default(add_28)
    add_29: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_28, 3)
    clamp_min_4: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_29, 0);  add_29 = None
    clamp_max_4: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_39: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_28, clamp_max_4);  add_28 = clamp_max_4 = None
    div_4: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_39, 6);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_4, primals_62, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 32, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_5: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_40: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_41: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_42: "f32[32]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
    add_32: "f32[32]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_17: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_43: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_44: "f32[32]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[32]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
    add_33: "f32[32]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    unsqueeze_20: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_46: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_21);  mul_40 = unsqueeze_21 = None
    unsqueeze_22: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_34: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_23);  mul_46 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 32, 56, 56]" = torch.ops.aten.clone.default(add_34)
    add_35: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_34, 3)
    clamp_min_5: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_35, 0);  add_35 = None
    clamp_max_5: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_47: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_34, clamp_max_5);  add_34 = clamp_max_5 = None
    div_5: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_47, 6);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_6: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_5, primals_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_112, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 32, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 32, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_37: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_6: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_48: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[32]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
    add_38: "f32[32]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_51: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_52: "f32[32]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[32]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
    add_39: "f32[32]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    unsqueeze_24: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_54: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_25);  mul_48 = unsqueeze_25 = None
    unsqueeze_26: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_40: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_27);  mul_54 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 32, 56, 56]" = torch.ops.aten.clone.default(add_40)
    add_41: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_40, 3)
    clamp_min_6: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max_6: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_55: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_40, clamp_max_6);  add_40 = clamp_max_6 = None
    div_6: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_55, 6);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(div_6, primals_64, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_115, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 32, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 32, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_43: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_7: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_56: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[32]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
    add_44: "f32[32]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_59: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
    mul_60: "f32[32]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[32]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
    add_45: "f32[32]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_28: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_62: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_29);  mul_56 = unsqueeze_29 = None
    unsqueeze_30: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_46: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_31);  mul_62 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 32, 28, 28]" = torch.ops.aten.clone.default(add_46)
    add_47: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_46, 3)
    clamp_min_7: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_min.default(add_47, 0);  add_47 = None
    clamp_max_7: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_63: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_46, clamp_max_7);  add_46 = clamp_max_7 = None
    div_7: "f32[8, 32, 28, 28]" = torch.ops.aten.div.Tensor(mul_63, 6);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_8: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_7, primals_65, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_118, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_49: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_8: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_64: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
    add_50: "f32[64]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
    add_51: "f32[64]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_70: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_33);  mul_64 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_52: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_35);  mul_70 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 64, 28, 28]" = torch.ops.aten.clone.default(add_52)
    add_53: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_52, 3)
    clamp_min_8: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_53, 0);  add_53 = None
    clamp_max_8: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_71: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_52, clamp_max_8);  add_52 = clamp_max_8 = None
    div_8: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_71, 6);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_8, primals_66, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_54: "i64[]" = torch.ops.aten.add.Tensor(primals_121, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_55: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_9: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_72: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
    add_56: "f32[64]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
    mul_76: "f32[64]" = torch.ops.aten.mul.Tensor(mul_75, 0.1);  mul_75 = None
    mul_77: "f32[64]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
    add_57: "f32[64]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_78: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_37);  mul_72 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_58: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_39);  mul_78 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 64, 28, 28]" = torch.ops.aten.clone.default(add_58)
    add_59: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_58, 3)
    clamp_min_9: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_59, 0);  add_59 = None
    clamp_max_9: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_79: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_58, clamp_max_9);  add_58 = clamp_max_9 = None
    div_9: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_79, 6);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_10: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_9, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_124, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 64, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 64, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_61: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_10: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_80: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_81: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_82: "f32[64]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
    add_62: "f32[64]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    squeeze_32: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_83: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_84: "f32[64]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
    mul_85: "f32[64]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
    add_63: "f32[64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_86: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_41);  mul_80 = unsqueeze_41 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_64: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_43);  mul_86 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 64, 28, 28]" = torch.ops.aten.clone.default(add_64)
    add_65: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_64, 3)
    clamp_min_10: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_65, 0);  add_65 = None
    clamp_max_10: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_87: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_64, clamp_max_10);  add_64 = clamp_max_10 = None
    div_10: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_87, 6);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(div_10, primals_68, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_66: "i64[]" = torch.ops.aten.add.Tensor(primals_127, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 64, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 64, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_67: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_11: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_88: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
    add_68: "f32[64]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_35: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_92: "f32[64]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[64]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
    add_69: "f32[64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_94: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_45);  mul_88 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_70: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_47);  mul_94 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 64, 14, 14]" = torch.ops.aten.clone.default(add_70)
    add_71: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_70, 3)
    clamp_min_11: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_min.default(add_71, 0);  add_71 = None
    clamp_max_11: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_95: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_70, clamp_max_11);  add_70 = clamp_max_11 = None
    div_11: "f32[8, 64, 14, 14]" = torch.ops.aten.div.Tensor(mul_95, 6);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_12: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_11, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_130, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_73: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_12: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_96: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_97: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_98: "f32[128]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
    add_74: "f32[128]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_99: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_100: "f32[128]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[128]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_102: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_49);  mul_96 = unsqueeze_49 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_76: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_51);  mul_102 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_76)
    add_77: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_76, 3)
    clamp_min_12: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_77, 0);  add_77 = None
    clamp_max_12: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_103: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_76, clamp_max_12);  add_76 = clamp_max_12 = None
    div_12: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_103, 6);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_12, primals_70, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_133, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_79: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_13: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_104: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_105: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(mul_107, 0.1);  mul_107 = None
    mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
    add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_110: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_53);  mul_104 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_82: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_55);  mul_110 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_82)
    add_83: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_82, 3)
    clamp_min_13: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_83, 0);  add_83 = None
    clamp_max_13: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_111: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_82, clamp_max_13);  add_82 = clamp_max_13 = None
    div_13: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_111, 6);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_14: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_13, primals_71, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_136, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_85: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_14: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_112: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
    add_86: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
    add_87: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_118: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_57);  mul_112 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_88: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_59);  mul_118 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_14: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_88)
    add_89: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_88, 3)
    clamp_min_14: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_89, 0);  add_89 = None
    clamp_max_14: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    mul_119: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_88, clamp_max_14);  add_88 = clamp_max_14 = None
    div_14: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_119, 6);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_14, primals_72, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_139, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_91: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_15: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_120: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_121: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
    add_92: "f32[128]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_123: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
    mul_125: "f32[128]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
    add_93: "f32[128]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_126: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_61);  mul_120 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_94: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_63);  mul_126 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_94)
    add_95: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_94, 3)
    clamp_min_15: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_95, 0);  add_95 = None
    clamp_max_15: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_127: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_94, clamp_max_15);  add_94 = clamp_max_15 = None
    div_15: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_127, 6);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_16: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_15, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_96: "i64[]" = torch.ops.aten.add.Tensor(primals_142, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_97: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_16: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_128: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_130: "f32[128]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
    add_98: "f32[128]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    squeeze_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_132: "f32[128]" = torch.ops.aten.mul.Tensor(mul_131, 0.1);  mul_131 = None
    mul_133: "f32[128]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
    add_99: "f32[128]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_134: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_65);  mul_128 = unsqueeze_65 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_100: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_67);  mul_134 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_100)
    add_101: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_100, 3)
    clamp_min_16: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_101, 0);  add_101 = None
    clamp_max_16: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_135: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_100, clamp_max_16);  add_100 = clamp_max_16 = None
    div_16: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_135, 6);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_16, primals_74, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_145, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_103: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_17: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_136: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
    add_104: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_140: "f32[128]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
    add_105: "f32[128]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_142: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_69);  mul_136 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_106: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_71);  mul_142 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_17: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_106)
    add_107: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_106, 3)
    clamp_min_17: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_107, 0);  add_107 = None
    clamp_max_17: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_143: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_106, clamp_max_17);  add_106 = clamp_max_17 = None
    div_17: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_143, 6);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_18: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_17, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_148, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_109: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_18: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_144: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_146: "f32[128]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
    add_110: "f32[128]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(mul_147, 0.1);  mul_147 = None
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_111: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_150: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_73);  mul_144 = unsqueeze_73 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_112: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_75);  mul_150 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_112)
    add_113: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_112, 3)
    clamp_min_18: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
    clamp_max_18: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_151: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_112, clamp_max_18);  add_112 = clamp_max_18 = None
    div_18: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_151, 6);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_18, primals_76, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_114: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_115: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_19: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
    mul_152: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_154: "f32[128]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
    add_116: "f32[128]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(mul_155, 0.1);  mul_155 = None
    mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_117: "f32[128]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_158: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_77);  mul_152 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_118: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_79);  mul_158 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_118)
    add_119: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_118, 3)
    clamp_min_19: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
    clamp_max_19: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_159: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_118, clamp_max_19);  add_118 = clamp_max_19 = None
    div_19: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_159, 6);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_19, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_154, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_121: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_20: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
    mul_160: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
    add_122: "f32[128]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    squeeze_62: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_123: "f32[128]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_166: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_81);  mul_160 = unsqueeze_81 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_124: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_83);  mul_166 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_20: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_124)
    add_125: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_124, 3)
    clamp_min_20: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_125, 0);  add_125 = None
    clamp_max_20: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_167: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_124, clamp_max_20);  add_124 = clamp_max_20 = None
    div_20: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_167, 6);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_20, primals_78, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_157, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_127: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_21: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
    mul_168: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_170: "f32[128]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_172: "f32[128]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[128]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_174: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_85);  mul_168 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_130: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_87);  mul_174 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_130)
    add_131: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_130, 3)
    clamp_min_21: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_131, 0);  add_131 = None
    clamp_max_21: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_175: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_130, clamp_max_21);  add_130 = clamp_max_21 = None
    div_21: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_175, 6);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_22: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_21, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_160, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_133: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_22: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
    mul_176: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
    add_134: "f32[128]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
    mul_181: "f32[128]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_135: "f32[128]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_182: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_89);  mul_176 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_136: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_91);  mul_182 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 128, 14, 14]" = torch.ops.aten.clone.default(add_136)
    add_137: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_136, 3)
    clamp_min_22: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_137, 0);  add_137 = None
    clamp_max_22: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_183: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, clamp_max_22);  add_136 = clamp_max_22 = None
    div_22: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_183, 6);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(div_22, primals_80, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_163, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_139: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_23: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
    mul_184: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
    add_140: "f32[128]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_188: "f32[128]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[128]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_141: "f32[128]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_190: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_93);  mul_184 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_142: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_95);  mul_190 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_23: "f32[8, 128, 7, 7]" = torch.ops.aten.clone.default(add_142)
    add_143: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(add_142, 3)
    clamp_min_23: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_min.default(add_143, 0);  add_143 = None
    clamp_max_23: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_191: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, clamp_max_23);  add_142 = clamp_max_23 = None
    div_23: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Tensor(mul_191, 6);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(div_23, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_24: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_81, primals_82, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_25: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_83, primals_84, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_144: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(convolution_25, 3)
    clamp_min_24: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_min.default(add_144, 0);  add_144 = None
    clamp_max_24: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    div_24: "f32[8, 128, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_24, 6);  clamp_max_24 = None
    mul_192: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(div_23, div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_26: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_192, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_166, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 256, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 256, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_146: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_24: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_49)
    mul_193: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_194: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_195: "f32[256]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
    add_147: "f32[256]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(mul_196, 0.1);  mul_196 = None
    mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_148: "f32[256]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_199: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_97);  mul_193 = unsqueeze_97 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_149: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_99);  mul_199 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 256, 7, 7]" = torch.ops.aten.clone.default(add_149)
    add_150: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_149, 3)
    clamp_min_25: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_150, 0);  add_150 = None
    clamp_max_25: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_200: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_149, clamp_max_25);  add_149 = clamp_max_25 = None
    div_25: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_200, 6);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_27: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(div_25, primals_86, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_169, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 256, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 256, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_152: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_25: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_51)
    mul_201: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_203: "f32[256]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
    add_153: "f32[256]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    squeeze_77: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(mul_204, 0.1);  mul_204 = None
    mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_154: "f32[256]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_207: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_101);  mul_201 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_155: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_103);  mul_207 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 256, 7, 7]" = torch.ops.aten.clone.default(add_155)
    add_156: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_155, 3)
    clamp_min_26: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
    clamp_max_26: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    mul_208: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_155, clamp_max_26);  add_155 = clamp_max_26 = None
    div_26: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_208, 6);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_28: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_87, primals_88, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_1: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_28);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_29: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_89, primals_90, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_157: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_29, 3)
    clamp_min_27: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_157, 0);  add_157 = None
    clamp_max_27: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    div_27: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_27, 6);  clamp_max_27 = None
    mul_209: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(div_26, div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_209, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_172, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 256, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 256, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_159: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_26: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_53)
    mul_210: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_211: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
    add_160: "f32[256]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_80: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_214: "f32[256]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[256]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_161: "f32[256]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_216: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_105);  mul_210 = unsqueeze_105 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_162: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_107);  mul_216 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_26: "f32[8, 256, 7, 7]" = torch.ops.aten.clone.default(add_162)
    add_163: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_162, 3)
    clamp_min_28: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_163, 0);  add_163 = None
    clamp_max_28: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_217: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_162, clamp_max_28);  add_162 = clamp_max_28 = None
    div_28: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_217, 6);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_2: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_28, [-1, -2], True);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_31: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    clone_27: "f32[8, 1280, 1, 1]" = torch.ops.aten.clone.default(convolution_31)
    add_164: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_31, 3)
    clamp_min_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_164, 0);  add_164 = None
    clamp_max_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_218: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_31, clamp_max_29);  convolution_31 = clamp_max_29 = None
    div_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_218, 6);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_1: "f32[8, 1280]" = torch.ops.aten.view.default(div_29, [8, 1280]);  div_29 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_56, view_1, permute);  primals_56 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    view_3: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    lt: "b8[8, 1280, 1, 1]" = torch.ops.aten.lt.Scalar(clone_27, -3)
    le: "b8[8, 1280, 1, 1]" = torch.ops.aten.le.Scalar(clone_27, 3)
    div_30: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(clone_27, 3);  clone_27 = None
    add_165: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div_30, 0.5);  div_30 = None
    mul_219: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, add_165);  add_165 = None
    where: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(le, mul_219, view_3);  le = mul_219 = view_3 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = scalar_tensor = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(where_1, mean_2, primals_92, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = mean_2 = primals_92 = None
    getitem_54: "f32[8, 256, 1, 1]" = convolution_backward[0]
    getitem_55: "f32[1280, 256, 1, 1]" = convolution_backward[1]
    getitem_56: "f32[1280]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 256, 7, 7]" = torch.ops.aten.expand.default(getitem_54, [8, 256, 7, 7]);  getitem_54 = None
    div_31: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_1: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_26, -3)
    le_1: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_26, 3)
    div_32: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_26, 3);  clone_26 = None
    add_166: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_32, 0.5);  div_32 = None
    mul_220: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(div_31, add_166);  add_166 = None
    where_2: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_1, mul_220, div_31);  le_1 = mul_220 = div_31 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_1, scalar_tensor_1, where_2);  lt_1 = scalar_tensor_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_108: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_109: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
    unsqueeze_110: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
    sum_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_27: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_110)
    mul_221: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_27);  sub_27 = None
    sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_111: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
    unsqueeze_112: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    unsqueeze_113: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
    mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    unsqueeze_114: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_115: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
    unsqueeze_116: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
    mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
    unsqueeze_118: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    unsqueeze_119: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
    sub_28: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_110);  convolution_30 = unsqueeze_110 = None
    mul_227: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_116);  sub_28 = unsqueeze_116 = None
    sub_29: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_227);  where_3 = mul_227 = None
    sub_30: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_113);  sub_29 = unsqueeze_113 = None
    mul_228: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_119);  sub_30 = unsqueeze_119 = None
    mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_79);  sum_3 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_228, mul_209, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_228 = mul_209 = primals_91 = None
    getitem_57: "f32[8, 256, 7, 7]" = convolution_backward_1[0]
    getitem_58: "f32[256, 256, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_230: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, div_26);  div_26 = None
    mul_231: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, div_27);  getitem_57 = div_27 = None
    sum_4: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2, 3], True);  mul_230 = None
    gt: "b8[8, 256, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_29, -3.0)
    lt_2: "b8[8, 256, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_29, 3.0);  convolution_29 = None
    bitwise_and: "b8[8, 256, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    mul_232: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_4, 0.16666666666666666);  sum_4 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_232, scalar_tensor_2);  bitwise_and = mul_232 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, relu_1, primals_89, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = primals_89 = None
    getitem_60: "f32[8, 64, 1, 1]" = convolution_backward_2[0]
    getitem_61: "f32[256, 64, 1, 1]" = convolution_backward_2[1]
    getitem_62: "f32[256]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_3: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_4: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    le_2: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(alias_4, 0);  alias_4 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_3, getitem_60);  le_2 = scalar_tensor_3 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_1, primals_87, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = mean_1 = primals_87 = None
    getitem_63: "f32[8, 256, 1, 1]" = convolution_backward_3[0]
    getitem_64: "f32[64, 256, 1, 1]" = convolution_backward_3[1]
    getitem_65: "f32[64]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 256, 7, 7]" = torch.ops.aten.expand.default(getitem_63, [8, 256, 7, 7]);  getitem_63 = None
    div_33: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_167: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_231, div_33);  mul_231 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_3: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_25, -3)
    le_3: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_25, 3)
    div_34: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_25, 3);  clone_25 = None
    add_168: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
    mul_233: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_167, add_168);  add_168 = None
    where_6: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_3, mul_233, add_167);  le_3 = mul_233 = add_167 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_3, scalar_tensor_4, where_6);  lt_3 = scalar_tensor_4 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_121: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
    unsqueeze_122: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_31: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_122)
    mul_234: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_31);  sub_31 = None
    sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
    mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_123: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
    unsqueeze_124: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    unsqueeze_125: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
    mul_236: "f32[256]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_237: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_238: "f32[256]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_127: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
    unsqueeze_128: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
    mul_239: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_129: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
    unsqueeze_130: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    unsqueeze_131: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
    sub_32: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_122);  convolution_27 = unsqueeze_122 = None
    mul_240: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_128);  sub_32 = unsqueeze_128 = None
    sub_33: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_240);  where_7 = mul_240 = None
    sub_34: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_125);  sub_33 = unsqueeze_125 = None
    mul_241: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_131);  sub_34 = unsqueeze_131 = None
    mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_76);  sum_6 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_241, div_25, primals_86, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_241 = div_25 = primals_86 = None
    getitem_66: "f32[8, 256, 7, 7]" = convolution_backward_4[0]
    getitem_67: "f32[256, 1, 5, 5]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_4: "b8[8, 256, 7, 7]" = torch.ops.aten.lt.Scalar(clone_24, -3)
    le_4: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(clone_24, 3)
    div_35: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(clone_24, 3);  clone_24 = None
    add_169: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(div_35, 0.5);  div_35 = None
    mul_243: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_66, add_169);  add_169 = None
    where_8: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_4, mul_243, getitem_66);  le_4 = mul_243 = getitem_66 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(lt_4, scalar_tensor_5, where_8);  lt_4 = scalar_tensor_5 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_132: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_133: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
    unsqueeze_134: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
    sum_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_35: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_134)
    mul_244: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_35);  sub_35 = None
    sum_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 2, 3]);  mul_244 = None
    mul_245: "f32[256]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
    unsqueeze_136: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    unsqueeze_137: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
    mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_247: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    unsqueeze_138: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
    unsqueeze_139: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
    unsqueeze_140: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
    mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_141: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
    unsqueeze_142: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    unsqueeze_143: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
    sub_36: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_134);  convolution_26 = unsqueeze_134 = None
    mul_250: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_140);  sub_36 = unsqueeze_140 = None
    sub_37: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_250);  where_9 = mul_250 = None
    sub_38: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_137);  sub_37 = unsqueeze_137 = None
    mul_251: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_143);  sub_38 = unsqueeze_143 = None
    mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_73);  sum_8 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_251, mul_192, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_251 = mul_192 = primals_85 = None
    getitem_69: "f32[8, 128, 7, 7]" = convolution_backward_5[0]
    getitem_70: "f32[256, 128, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_253: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_69, div_23);  div_23 = None
    mul_254: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_69, div_24);  getitem_69 = div_24 = None
    sum_9: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2, 3], True);  mul_253 = None
    gt_1: "b8[8, 128, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_25, -3.0)
    lt_5: "b8[8, 128, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_25, 3.0);  convolution_25 = None
    bitwise_and_1: "b8[8, 128, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_5);  gt_1 = lt_5 = None
    mul_255: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_9, 0.16666666666666666);  sum_9 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 128, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_255, scalar_tensor_6);  bitwise_and_1 = mul_255 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_10, relu, primals_83, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_10 = primals_83 = None
    getitem_72: "f32[8, 32, 1, 1]" = convolution_backward_6[0]
    getitem_73: "f32[128, 32, 1, 1]" = convolution_backward_6[1]
    getitem_74: "f32[128]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_6: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_7: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    le_5: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_7, 0);  alias_7 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_7, getitem_72);  le_5 = scalar_tensor_7 = getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_81, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_11 = mean = primals_81 = None
    getitem_75: "f32[8, 128, 1, 1]" = convolution_backward_7[0]
    getitem_76: "f32[32, 128, 1, 1]" = convolution_backward_7[1]
    getitem_77: "f32[32]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 128, 7, 7]" = torch.ops.aten.expand.default(getitem_75, [8, 128, 7, 7]);  getitem_75 = None
    div_36: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_170: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_254, div_36);  mul_254 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_6: "b8[8, 128, 7, 7]" = torch.ops.aten.lt.Scalar(clone_23, -3)
    le_6: "b8[8, 128, 7, 7]" = torch.ops.aten.le.Scalar(clone_23, 3)
    div_37: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Tensor(clone_23, 3);  clone_23 = None
    add_171: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
    mul_256: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(add_170, add_171);  add_171 = None
    where_12: "f32[8, 128, 7, 7]" = torch.ops.aten.where.self(le_6, mul_256, add_170);  le_6 = mul_256 = add_170 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 128, 7, 7]" = torch.ops.aten.where.self(lt_6, scalar_tensor_8, where_12);  lt_6 = scalar_tensor_8 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_145: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
    unsqueeze_146: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
    sum_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_39: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_146)
    mul_257: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_39);  sub_39 = None
    sum_11: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
    mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_147: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
    unsqueeze_148: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    unsqueeze_149: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    unsqueeze_150: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
    unsqueeze_151: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
    unsqueeze_152: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
    mul_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_153: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
    unsqueeze_154: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    unsqueeze_155: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
    sub_40: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_146);  convolution_23 = unsqueeze_146 = None
    mul_263: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_152);  sub_40 = unsqueeze_152 = None
    sub_41: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_263);  where_13 = mul_263 = None
    sub_42: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_149);  sub_41 = unsqueeze_149 = None
    mul_264: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_155);  sub_42 = unsqueeze_155 = None
    mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_70);  sum_11 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_264, div_22, primals_80, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_264 = div_22 = primals_80 = None
    getitem_78: "f32[8, 128, 14, 14]" = convolution_backward_8[0]
    getitem_79: "f32[128, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_7: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_22, -3)
    le_7: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_22, 3)
    div_38: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_22, 3);  clone_22 = None
    add_172: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_38, 0.5);  div_38 = None
    mul_266: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, add_172);  add_172 = None
    where_14: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_7, mul_266, getitem_78);  le_7 = mul_266 = getitem_78 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_7, scalar_tensor_9, where_14);  lt_7 = scalar_tensor_9 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_156: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_157: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    sum_12: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_43: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_158)
    mul_267: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_43);  sub_43 = None
    sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
    mul_268: "f32[128]" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
    unsqueeze_159: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_160: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    unsqueeze_161: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
    mul_269: "f32[128]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    mul_270: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_271: "f32[128]" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    unsqueeze_162: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_163: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
    unsqueeze_164: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
    mul_272: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_165: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_166: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    sub_44: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_158);  convolution_22 = unsqueeze_158 = None
    mul_273: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_164);  sub_44 = unsqueeze_164 = None
    sub_45: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_273);  where_15 = mul_273 = None
    sub_46: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_161);  sub_45 = unsqueeze_161 = None
    mul_274: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_167);  sub_46 = unsqueeze_167 = None
    mul_275: "f32[128]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_67);  sum_13 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_274, div_21, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_274 = div_21 = primals_79 = None
    getitem_81: "f32[8, 128, 14, 14]" = convolution_backward_9[0]
    getitem_82: "f32[128, 128, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_8: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_21, -3)
    le_8: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_21, 3)
    div_39: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_21, 3);  clone_21 = None
    add_173: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
    mul_276: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_81, add_173);  add_173 = None
    where_16: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_8, mul_276, getitem_81);  le_8 = mul_276 = getitem_81 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_8, scalar_tensor_10, where_16);  lt_8 = scalar_tensor_10 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_169: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_47: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_170)
    mul_277: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_47);  sub_47 = None
    sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3]);  mul_277 = None
    mul_278: "f32[128]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    unsqueeze_171: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_172: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    unsqueeze_173: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
    mul_279: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
    mul_280: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_281: "f32[128]" = torch.ops.aten.mul.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    unsqueeze_174: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_175: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
    unsqueeze_176: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
    mul_282: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_177: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
    unsqueeze_178: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    sub_48: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_170);  convolution_21 = unsqueeze_170 = None
    mul_283: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_176);  sub_48 = unsqueeze_176 = None
    sub_49: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_283);  where_17 = mul_283 = None
    sub_50: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_173);  sub_49 = unsqueeze_173 = None
    mul_284: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_179);  sub_50 = unsqueeze_179 = None
    mul_285: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_64);  sum_15 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_284, div_20, primals_78, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_284 = div_20 = primals_78 = None
    getitem_84: "f32[8, 128, 14, 14]" = convolution_backward_10[0]
    getitem_85: "f32[128, 1, 5, 5]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_9: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le_9: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_40: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_174: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
    mul_286: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_84, add_174);  add_174 = None
    where_18: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_9, mul_286, getitem_84);  le_9 = mul_286 = getitem_84 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_9, scalar_tensor_11, where_18);  lt_9 = scalar_tensor_11 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_180: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_181: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_51: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_182)
    mul_287: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_51);  sub_51 = None
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_288: "f32[128]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
    unsqueeze_183: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
    unsqueeze_184: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    unsqueeze_185: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
    mul_289: "f32[128]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_290: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_291: "f32[128]" = torch.ops.aten.mul.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    unsqueeze_186: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_187: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
    unsqueeze_188: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
    mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_189: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_190: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    sub_52: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_182);  convolution_20 = unsqueeze_182 = None
    mul_293: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_188);  sub_52 = unsqueeze_188 = None
    sub_53: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_293);  where_19 = mul_293 = None
    sub_54: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_185);  sub_53 = unsqueeze_185 = None
    mul_294: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_191);  sub_54 = unsqueeze_191 = None
    mul_295: "f32[128]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_61);  sum_17 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_294, div_19, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_294 = div_19 = primals_77 = None
    getitem_87: "f32[8, 128, 14, 14]" = convolution_backward_11[0]
    getitem_88: "f32[128, 128, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_10: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_10: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_41: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_175: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_41, 0.5);  div_41 = None
    mul_296: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, add_175);  add_175 = None
    where_20: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_10, mul_296, getitem_87);  le_10 = mul_296 = getitem_87 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_10, scalar_tensor_12, where_20);  lt_10 = scalar_tensor_12 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_192: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_193: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    sum_18: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_55: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_194)
    mul_297: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_55);  sub_55 = None
    sum_19: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_298: "f32[128]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_195: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_196: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    unsqueeze_197: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
    mul_299: "f32[128]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_300: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_301: "f32[128]" = torch.ops.aten.mul.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    unsqueeze_198: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_199: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
    unsqueeze_200: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
    mul_302: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_201: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_202: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    sub_56: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_194);  convolution_19 = unsqueeze_194 = None
    mul_303: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_200);  sub_56 = unsqueeze_200 = None
    sub_57: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_303);  where_21 = mul_303 = None
    sub_58: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_197);  sub_57 = unsqueeze_197 = None
    mul_304: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_203);  sub_58 = unsqueeze_203 = None
    mul_305: "f32[128]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_58);  sum_19 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_304, div_18, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_304 = div_18 = primals_76 = None
    getitem_90: "f32[8, 128, 14, 14]" = convolution_backward_12[0]
    getitem_91: "f32[128, 1, 5, 5]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_11: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_11: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_42: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_176: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
    mul_306: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_90, add_176);  add_176 = None
    where_22: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_11, mul_306, getitem_90);  le_11 = mul_306 = getitem_90 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_11, scalar_tensor_13, where_22);  lt_11 = scalar_tensor_13 = where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_59: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_206)
    mul_307: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_59);  sub_59 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 2, 3]);  mul_307 = None
    mul_308: "f32[128]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_207: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_208: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    unsqueeze_209: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
    mul_309: "f32[128]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_310: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_311: "f32[128]" = torch.ops.aten.mul.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    unsqueeze_210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_311, 0);  mul_311 = None
    unsqueeze_211: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
    unsqueeze_212: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
    mul_312: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
    unsqueeze_214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    sub_60: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_206);  convolution_18 = unsqueeze_206 = None
    mul_313: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_212);  sub_60 = unsqueeze_212 = None
    sub_61: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_313);  where_23 = mul_313 = None
    sub_62: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_209);  sub_61 = unsqueeze_209 = None
    mul_314: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_215);  sub_62 = unsqueeze_215 = None
    mul_315: "f32[128]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_55);  sum_21 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_314, div_17, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_314 = div_17 = primals_75 = None
    getitem_93: "f32[8, 128, 14, 14]" = convolution_backward_13[0]
    getitem_94: "f32[128, 128, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_12: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_12: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_43: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_177: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
    mul_316: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, add_177);  add_177 = None
    where_24: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_12, mul_316, getitem_93);  le_12 = mul_316 = getitem_93 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_12, scalar_tensor_14, where_24);  lt_12 = scalar_tensor_14 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_216: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_217: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_63: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_218)
    mul_317: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_63);  sub_63 = None
    sum_23: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 2, 3]);  mul_317 = None
    mul_318: "f32[128]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_219: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_220: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    unsqueeze_221: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
    mul_319: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_320: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_321: "f32[128]" = torch.ops.aten.mul.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_222: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_223: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    mul_322: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_225: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_226: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    sub_64: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_218);  convolution_17 = unsqueeze_218 = None
    mul_323: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_224);  sub_64 = unsqueeze_224 = None
    sub_65: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_323);  where_25 = mul_323 = None
    sub_66: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_221);  sub_65 = unsqueeze_221 = None
    mul_324: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_227);  sub_66 = unsqueeze_227 = None
    mul_325: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_52);  sum_23 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_324, div_16, primals_74, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_324 = div_16 = primals_74 = None
    getitem_96: "f32[8, 128, 14, 14]" = convolution_backward_14[0]
    getitem_97: "f32[128, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_13: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_13: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_44: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_178: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_44, 0.5);  div_44 = None
    mul_326: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_96, add_178);  add_178 = None
    where_26: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_13, mul_326, getitem_96);  le_13 = mul_326 = getitem_96 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_13, scalar_tensor_15, where_26);  lt_13 = scalar_tensor_15 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_228: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_229: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_67: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_230)
    mul_327: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_67);  sub_67 = None
    sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_328: "f32[128]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_231: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_232: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_329: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_330: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_331: "f32[128]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_235: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    mul_332: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_237: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_238: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    sub_68: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_230);  convolution_16 = unsqueeze_230 = None
    mul_333: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_236);  sub_68 = unsqueeze_236 = None
    sub_69: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_333);  where_27 = mul_333 = None
    sub_70: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_233);  sub_69 = unsqueeze_233 = None
    mul_334: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_239);  sub_70 = unsqueeze_239 = None
    mul_335: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_49);  sum_25 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_334, div_15, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = div_15 = primals_73 = None
    getitem_99: "f32[8, 128, 14, 14]" = convolution_backward_15[0]
    getitem_100: "f32[128, 128, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_14: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_14: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_45: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_179: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
    mul_336: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_99, add_179);  add_179 = None
    where_28: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_14, mul_336, getitem_99);  le_14 = mul_336 = getitem_99 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_14, scalar_tensor_16, where_28);  lt_14 = scalar_tensor_16 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_71: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_242)
    mul_337: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_71);  sub_71 = None
    sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[128]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_244: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_339: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_341: "f32[128]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_246: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_247: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    mul_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_250: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    sub_72: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_242);  convolution_15 = unsqueeze_242 = None
    mul_343: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_248);  sub_72 = unsqueeze_248 = None
    sub_73: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_343);  where_29 = mul_343 = None
    sub_74: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_245);  sub_73 = unsqueeze_245 = None
    mul_344: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_251);  sub_74 = unsqueeze_251 = None
    mul_345: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_46);  sum_27 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_344, div_14, primals_72, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_344 = div_14 = primals_72 = None
    getitem_102: "f32[8, 128, 14, 14]" = convolution_backward_16[0]
    getitem_103: "f32[128, 1, 5, 5]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_15: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_15: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_46: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_180: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
    mul_346: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_102, add_180);  add_180 = None
    where_30: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_15, mul_346, getitem_102);  le_15 = mul_346 = getitem_102 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_15, scalar_tensor_17, where_30);  lt_15 = scalar_tensor_17 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_253: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    sum_28: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_75: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_254)
    mul_347: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_75);  sub_75 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 2, 3]);  mul_347 = None
    mul_348: "f32[128]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_255: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_256: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_350: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
    unsqueeze_258: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_259: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_261: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
    unsqueeze_262: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    sub_76: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_254);  convolution_14 = unsqueeze_254 = None
    mul_353: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_260);  sub_76 = unsqueeze_260 = None
    sub_77: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_353);  where_31 = mul_353 = None
    sub_78: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_257);  sub_77 = unsqueeze_257 = None
    mul_354: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_263);  sub_78 = unsqueeze_263 = None
    mul_355: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_43);  sum_29 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_354, div_13, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = div_13 = primals_71 = None
    getitem_105: "f32[8, 128, 14, 14]" = convolution_backward_17[0]
    getitem_106: "f32[128, 128, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_16: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_16: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_47: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_181: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
    mul_356: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_105, add_181);  add_181 = None
    where_32: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_16, mul_356, getitem_105);  le_16 = mul_356 = getitem_105 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_16, scalar_tensor_18, where_32);  lt_16 = scalar_tensor_18 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_264: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_265: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_79: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_266)
    mul_357: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_79);  sub_79 = None
    sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_358: "f32[128]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_267: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_268: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_359: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_360: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_361: "f32[128]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    unsqueeze_270: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_271: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    mul_362: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_273: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_274: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    sub_80: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_266);  convolution_13 = unsqueeze_266 = None
    mul_363: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_272);  sub_80 = unsqueeze_272 = None
    sub_81: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_363);  where_33 = mul_363 = None
    sub_82: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_269);  sub_81 = unsqueeze_269 = None
    mul_364: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_275);  sub_82 = unsqueeze_275 = None
    mul_365: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_40);  sum_31 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_364, div_12, primals_70, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_364 = div_12 = primals_70 = None
    getitem_108: "f32[8, 128, 14, 14]" = convolution_backward_18[0]
    getitem_109: "f32[128, 1, 5, 5]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_17: "b8[8, 128, 14, 14]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_17: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_48: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_182: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_366: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, add_182);  add_182 = None
    where_34: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_17, mul_366, getitem_108);  le_17 = mul_366 = getitem_108 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(lt_17, scalar_tensor_19, where_34);  lt_17 = scalar_tensor_19 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_276: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_277: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    sum_32: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_83: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_278)
    mul_367: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_83);  sub_83 = None
    sum_33: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 2, 3]);  mul_367 = None
    mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_279: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_280: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_369: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_370: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_371: "f32[128]" = torch.ops.aten.mul.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    unsqueeze_282: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_283: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_285: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_286: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    sub_84: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_278);  convolution_12 = unsqueeze_278 = None
    mul_373: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_284);  sub_84 = unsqueeze_284 = None
    sub_85: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_373);  where_35 = mul_373 = None
    sub_86: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_281);  sub_85 = unsqueeze_281 = None
    mul_374: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_287);  sub_86 = unsqueeze_287 = None
    mul_375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_37);  sum_33 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_374, div_11, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_374 = div_11 = primals_69 = None
    getitem_111: "f32[8, 64, 14, 14]" = convolution_backward_19[0]
    getitem_112: "f32[128, 64, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_18: "b8[8, 64, 14, 14]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_18: "b8[8, 64, 14, 14]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_49: "f32[8, 64, 14, 14]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_183: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_376: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_111, add_183);  add_183 = None
    where_36: "f32[8, 64, 14, 14]" = torch.ops.aten.where.self(le_18, mul_376, getitem_111);  le_18 = mul_376 = getitem_111 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[8, 64, 14, 14]" = torch.ops.aten.where.self(lt_18, scalar_tensor_20, where_36);  lt_18 = scalar_tensor_20 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_288: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_289: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    sum_34: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_87: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_290)
    mul_377: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_87);  sub_87 = None
    sum_35: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
    mul_378: "f32[64]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_291: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_292: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_379: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_380: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_381: "f32[64]" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_294: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
    unsqueeze_295: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    mul_382: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_297: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_298: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    sub_88: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_290);  convolution_11 = unsqueeze_290 = None
    mul_383: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_296);  sub_88 = unsqueeze_296 = None
    sub_89: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_383);  where_37 = mul_383 = None
    sub_90: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_293);  sub_89 = unsqueeze_293 = None
    mul_384: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_299);  sub_90 = unsqueeze_299 = None
    mul_385: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_34);  sum_35 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_384, div_10, primals_68, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_384 = div_10 = primals_68 = None
    getitem_114: "f32[8, 64, 28, 28]" = convolution_backward_20[0]
    getitem_115: "f32[64, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_19: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_19: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_50: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_184: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_386: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, add_184);  add_184 = None
    where_38: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_19, mul_386, getitem_114);  le_19 = mul_386 = getitem_114 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_19, scalar_tensor_21, where_38);  lt_19 = scalar_tensor_21 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_300: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_301: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    sum_36: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_91: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_302)
    mul_387: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_39, sub_91);  sub_91 = None
    sum_37: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3]);  mul_387 = None
    mul_388: "f32[64]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
    unsqueeze_303: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_304: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_389: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
    mul_390: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_391: "f32[64]" = torch.ops.aten.mul.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_306: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
    unsqueeze_307: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    mul_392: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_309: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_310: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    sub_92: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_302);  convolution_10 = unsqueeze_302 = None
    mul_393: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_308);  sub_92 = unsqueeze_308 = None
    sub_93: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_39, mul_393);  where_39 = mul_393 = None
    sub_94: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_305);  sub_93 = unsqueeze_305 = None
    mul_394: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_311);  sub_94 = unsqueeze_311 = None
    mul_395: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_31);  sum_37 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_394, div_9, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_394 = div_9 = primals_67 = None
    getitem_117: "f32[8, 64, 28, 28]" = convolution_backward_21[0]
    getitem_118: "f32[64, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_20: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_20: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_51: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_185: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_396: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_117, add_185);  add_185 = None
    where_40: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_20, mul_396, getitem_117);  le_20 = mul_396 = getitem_117 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_20, scalar_tensor_22, where_40);  lt_20 = scalar_tensor_22 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_312: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_313: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    sum_38: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_95: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_314)
    mul_397: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_41, sub_95);  sub_95 = None
    sum_39: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 2, 3]);  mul_397 = None
    mul_398: "f32[64]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
    unsqueeze_315: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_316: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_399: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
    mul_400: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_401: "f32[64]" = torch.ops.aten.mul.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    unsqueeze_318: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_319: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    mul_402: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_321: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
    unsqueeze_322: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    sub_96: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_314);  convolution_9 = unsqueeze_314 = None
    mul_403: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_320);  sub_96 = unsqueeze_320 = None
    sub_97: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_41, mul_403);  where_41 = mul_403 = None
    sub_98: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_317);  sub_97 = unsqueeze_317 = None
    mul_404: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_323);  sub_98 = unsqueeze_323 = None
    mul_405: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_28);  sum_39 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_404, div_8, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_404 = div_8 = primals_66 = None
    getitem_120: "f32[8, 64, 28, 28]" = convolution_backward_22[0]
    getitem_121: "f32[64, 1, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_21: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_21: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_52: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_186: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_406: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_120, add_186);  add_186 = None
    where_42: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_21, mul_406, getitem_120);  le_21 = mul_406 = getitem_120 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_21, scalar_tensor_23, where_42);  lt_21 = scalar_tensor_23 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_324: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_325: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    sum_40: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_99: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_326)
    mul_407: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_43, sub_99);  sub_99 = None
    sum_41: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_408: "f32[64]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
    unsqueeze_327: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_328: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_409: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
    mul_410: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_411: "f32[64]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    unsqueeze_330: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_331: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    mul_412: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_333: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_334: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    sub_100: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_326);  convolution_8 = unsqueeze_326 = None
    mul_413: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_332);  sub_100 = unsqueeze_332 = None
    sub_101: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_43, mul_413);  where_43 = mul_413 = None
    sub_102: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_329);  sub_101 = unsqueeze_329 = None
    mul_414: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_335);  sub_102 = unsqueeze_335 = None
    mul_415: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_25);  sum_41 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_414, div_7, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = div_7 = primals_65 = None
    getitem_123: "f32[8, 32, 28, 28]" = convolution_backward_23[0]
    getitem_124: "f32[64, 32, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_22: "b8[8, 32, 28, 28]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_22: "b8[8, 32, 28, 28]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_53: "f32[8, 32, 28, 28]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_187: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_416: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_123, add_187);  add_187 = None
    where_44: "f32[8, 32, 28, 28]" = torch.ops.aten.where.self(le_22, mul_416, getitem_123);  le_22 = mul_416 = getitem_123 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[8, 32, 28, 28]" = torch.ops.aten.where.self(lt_22, scalar_tensor_24, where_44);  lt_22 = scalar_tensor_24 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_336: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_337: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    sum_42: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_103: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_338)
    mul_417: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(where_45, sub_103);  sub_103 = None
    sum_43: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
    mul_418: "f32[32]" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
    unsqueeze_339: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_340: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_419: "f32[32]" = torch.ops.aten.mul.Tensor(sum_43, 0.00015943877551020407)
    mul_420: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_421: "f32[32]" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    unsqueeze_342: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_343: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    mul_422: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_345: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_346: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    sub_104: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_338);  convolution_7 = unsqueeze_338 = None
    mul_423: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_344);  sub_104 = unsqueeze_344 = None
    sub_105: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(where_45, mul_423);  where_45 = mul_423 = None
    sub_106: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_341);  sub_105 = unsqueeze_341 = None
    mul_424: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_347);  sub_106 = unsqueeze_347 = None
    mul_425: "f32[32]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_22);  sum_43 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_424, div_6, primals_64, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_424 = div_6 = primals_64 = None
    getitem_126: "f32[8, 32, 56, 56]" = convolution_backward_24[0]
    getitem_127: "f32[32, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_23: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_23: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_54: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_188: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_426: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_126, add_188);  add_188 = None
    where_46: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_23, mul_426, getitem_126);  le_23 = mul_426 = getitem_126 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_23, scalar_tensor_25, where_46);  lt_23 = scalar_tensor_25 = where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_348: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_349: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    sum_44: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_107: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_350)
    mul_427: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_107);  sub_107 = None
    sum_45: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[32]" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
    unsqueeze_351: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_352: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_429: "f32[32]" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
    mul_430: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_431: "f32[32]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_354: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_355: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_432: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_357: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_358: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    sub_108: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_350);  convolution_6 = unsqueeze_350 = None
    mul_433: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_356);  sub_108 = unsqueeze_356 = None
    sub_109: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_47, mul_433);  where_47 = mul_433 = None
    sub_110: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_353);  sub_109 = unsqueeze_353 = None
    mul_434: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_359);  sub_110 = unsqueeze_359 = None
    mul_435: "f32[32]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_19);  sum_45 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_434, div_5, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = div_5 = primals_63 = None
    getitem_129: "f32[8, 32, 56, 56]" = convolution_backward_25[0]
    getitem_130: "f32[32, 32, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_24: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_24: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_55: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_189: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_55, 0.5);  div_55 = None
    mul_436: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_129, add_189);  add_189 = None
    where_48: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_24, mul_436, getitem_129);  le_24 = mul_436 = getitem_129 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_24, scalar_tensor_26, where_48);  lt_24 = scalar_tensor_26 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_360: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_361: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    sum_46: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_111: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_362)
    mul_437: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_49, sub_111);  sub_111 = None
    sum_47: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_438: "f32[32]" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
    unsqueeze_363: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
    unsqueeze_364: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_439: "f32[32]" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
    mul_440: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_441: "f32[32]" = torch.ops.aten.mul.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    unsqueeze_366: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_367: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_442: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_369: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_370: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    sub_112: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_362);  convolution_5 = unsqueeze_362 = None
    mul_443: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_368);  sub_112 = unsqueeze_368 = None
    sub_113: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_49, mul_443);  where_49 = mul_443 = None
    sub_114: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_365);  sub_113 = unsqueeze_365 = None
    mul_444: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_371);  sub_114 = unsqueeze_371 = None
    mul_445: "f32[32]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_16);  sum_47 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_444, div_4, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_444 = div_4 = primals_62 = None
    getitem_132: "f32[8, 32, 56, 56]" = convolution_backward_26[0]
    getitem_133: "f32[32, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_25: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_25: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_56: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_190: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_56, 0.5);  div_56 = None
    mul_446: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_132, add_190);  add_190 = None
    where_50: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_25, mul_446, getitem_132);  le_25 = mul_446 = getitem_132 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_51: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_25, scalar_tensor_27, where_50);  lt_25 = scalar_tensor_27 = where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_372: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_373: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    sum_48: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_115: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_374)
    mul_447: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_51, sub_115);  sub_115 = None
    sum_49: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_448: "f32[32]" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
    unsqueeze_375: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_376: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_449: "f32[32]" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
    mul_450: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_451: "f32[32]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_378: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_379: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_452: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_381: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_382: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    sub_116: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_374);  convolution_4 = unsqueeze_374 = None
    mul_453: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_380);  sub_116 = unsqueeze_380 = None
    sub_117: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_51, mul_453);  where_51 = mul_453 = None
    sub_118: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_377);  sub_117 = unsqueeze_377 = None
    mul_454: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_383);  sub_118 = unsqueeze_383 = None
    mul_455: "f32[32]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_13);  sum_49 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_454, div_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_454 = div_3 = primals_61 = None
    getitem_135: "f32[8, 16, 56, 56]" = convolution_backward_27[0]
    getitem_136: "f32[32, 16, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_26: "b8[8, 16, 56, 56]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_26: "b8[8, 16, 56, 56]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_57: "f32[8, 16, 56, 56]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_191: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(div_57, 0.5);  div_57 = None
    mul_456: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_135, add_191);  add_191 = None
    where_52: "f32[8, 16, 56, 56]" = torch.ops.aten.where.self(le_26, mul_456, getitem_135);  le_26 = mul_456 = getitem_135 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_53: "f32[8, 16, 56, 56]" = torch.ops.aten.where.self(lt_26, scalar_tensor_28, where_52);  lt_26 = scalar_tensor_28 = where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_384: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_385: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    sum_50: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_119: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_386)
    mul_457: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(where_53, sub_119);  sub_119 = None
    sum_51: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
    mul_458: "f32[16]" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
    unsqueeze_387: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_388: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_459: "f32[16]" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
    mul_460: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_461: "f32[16]" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_390: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_391: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_462: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_393: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_394: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    sub_120: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_386);  convolution_3 = unsqueeze_386 = None
    mul_463: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_392);  sub_120 = unsqueeze_392 = None
    sub_121: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(where_53, mul_463);  where_53 = mul_463 = None
    sub_122: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_389);  sub_121 = unsqueeze_389 = None
    mul_464: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_395);  sub_122 = unsqueeze_395 = None
    mul_465: "f32[16]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_10);  sum_51 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_464, div_2, primals_60, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_464 = div_2 = primals_60 = None
    getitem_138: "f32[8, 16, 112, 112]" = convolution_backward_28[0]
    getitem_139: "f32[16, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_27: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_27: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_58: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_192: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_466: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_138, add_192);  add_192 = None
    where_54: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_27, mul_466, getitem_138);  le_27 = mul_466 = getitem_138 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_55: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_27, scalar_tensor_29, where_54);  lt_27 = scalar_tensor_29 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_397: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    sum_52: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_123: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_398)
    mul_467: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_55, sub_123);  sub_123 = None
    sum_53: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[16]" = torch.ops.aten.mul.Tensor(sum_52, 9.964923469387754e-06)
    unsqueeze_399: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_400: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_469: "f32[16]" = torch.ops.aten.mul.Tensor(sum_53, 9.964923469387754e-06)
    mul_470: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_471: "f32[16]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_402: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_403: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_472: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_405: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_406: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    sub_124: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_398);  convolution_2 = unsqueeze_398 = None
    mul_473: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_404);  sub_124 = unsqueeze_404 = None
    sub_125: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_55, mul_473);  where_55 = mul_473 = None
    sub_126: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_401);  sub_125 = unsqueeze_401 = None
    mul_474: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_407);  sub_126 = unsqueeze_407 = None
    mul_475: "f32[16]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_7);  sum_53 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_474, div_1, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = div_1 = primals_59 = None
    getitem_141: "f32[8, 8, 112, 112]" = convolution_backward_29[0]
    getitem_142: "f32[16, 8, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_28: "b8[8, 8, 112, 112]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_28: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_59: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_193: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(div_59, 0.5);  div_59 = None
    mul_476: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_141, add_193);  add_193 = None
    where_56: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_28, mul_476, getitem_141);  le_28 = mul_476 = getitem_141 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_57: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(lt_28, scalar_tensor_30, where_56);  lt_28 = scalar_tensor_30 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_409: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    sum_54: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_127: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_410)
    mul_477: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_57, sub_127);  sub_127 = None
    sum_55: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2, 3]);  mul_477 = None
    mul_478: "f32[8]" = torch.ops.aten.mul.Tensor(sum_54, 9.964923469387754e-06)
    unsqueeze_411: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_412: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_479: "f32[8]" = torch.ops.aten.mul.Tensor(sum_55, 9.964923469387754e-06)
    mul_480: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_481: "f32[8]" = torch.ops.aten.mul.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    unsqueeze_414: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_415: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_482: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_417: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_418: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    sub_128: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_410);  convolution_1 = unsqueeze_410 = None
    mul_483: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_416);  sub_128 = unsqueeze_416 = None
    sub_129: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_57, mul_483);  where_57 = mul_483 = None
    sub_130: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_413);  sub_129 = unsqueeze_413 = None
    mul_484: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_419);  sub_130 = unsqueeze_419 = None
    mul_485: "f32[8]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_4);  sum_55 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_484, div, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_484 = div = primals_58 = None
    getitem_144: "f32[8, 8, 112, 112]" = convolution_backward_30[0]
    getitem_145: "f32[8, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_29: "b8[8, 8, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_29: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_60: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_194: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(div_60, 0.5);  div_60 = None
    mul_486: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, add_194);  add_194 = None
    where_58: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_29, mul_486, getitem_144);  le_29 = mul_486 = getitem_144 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_59: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(lt_29, scalar_tensor_31, where_58);  lt_29 = scalar_tensor_31 = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_421: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_56: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_131: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_422)
    mul_487: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_59, sub_131);  sub_131 = None
    sum_57: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3]);  mul_487 = None
    mul_488: "f32[8]" = torch.ops.aten.mul.Tensor(sum_56, 9.964923469387754e-06)
    unsqueeze_423: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_424: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_489: "f32[8]" = torch.ops.aten.mul.Tensor(sum_57, 9.964923469387754e-06)
    mul_490: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_491: "f32[8]" = torch.ops.aten.mul.Tensor(mul_489, mul_490);  mul_489 = mul_490 = None
    unsqueeze_426: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_427: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_492: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_429: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_430: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    sub_132: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_422);  convolution = unsqueeze_422 = None
    mul_493: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_428);  sub_132 = unsqueeze_428 = None
    sub_133: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_59, mul_493);  where_59 = mul_493 = None
    sub_134: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_425);  sub_133 = unsqueeze_425 = None
    mul_494: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_431);  sub_134 = unsqueeze_431 = None
    mul_495: "f32[8]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_1);  sum_57 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_494, primals_175, primals_57, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_494 = primals_175 = primals_57 = None
    getitem_148: "f32[8, 3, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_94, add);  primals_94 = add = None
    copy__1: "f32[8]" = torch.ops.aten.copy_.default(primals_95, add_2);  primals_95 = add_2 = None
    copy__2: "f32[8]" = torch.ops.aten.copy_.default(primals_96, add_3);  primals_96 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_97, add_6);  primals_97 = add_6 = None
    copy__4: "f32[8]" = torch.ops.aten.copy_.default(primals_98, add_8);  primals_98 = add_8 = None
    copy__5: "f32[8]" = torch.ops.aten.copy_.default(primals_99, add_9);  primals_99 = add_9 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_100, add_12);  primals_100 = add_12 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_101, add_14);  primals_101 = add_14 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_102, add_15);  primals_102 = add_15 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_103, add_18);  primals_103 = add_18 = None
    copy__10: "f32[16]" = torch.ops.aten.copy_.default(primals_104, add_20);  primals_104 = add_20 = None
    copy__11: "f32[16]" = torch.ops.aten.copy_.default(primals_105, add_21);  primals_105 = add_21 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_106, add_24);  primals_106 = add_24 = None
    copy__13: "f32[32]" = torch.ops.aten.copy_.default(primals_107, add_26);  primals_107 = add_26 = None
    copy__14: "f32[32]" = torch.ops.aten.copy_.default(primals_108, add_27);  primals_108 = add_27 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_30);  primals_109 = add_30 = None
    copy__16: "f32[32]" = torch.ops.aten.copy_.default(primals_110, add_32);  primals_110 = add_32 = None
    copy__17: "f32[32]" = torch.ops.aten.copy_.default(primals_111, add_33);  primals_111 = add_33 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_112, add_36);  primals_112 = add_36 = None
    copy__19: "f32[32]" = torch.ops.aten.copy_.default(primals_113, add_38);  primals_113 = add_38 = None
    copy__20: "f32[32]" = torch.ops.aten.copy_.default(primals_114, add_39);  primals_114 = add_39 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_115, add_42);  primals_115 = add_42 = None
    copy__22: "f32[32]" = torch.ops.aten.copy_.default(primals_116, add_44);  primals_116 = add_44 = None
    copy__23: "f32[32]" = torch.ops.aten.copy_.default(primals_117, add_45);  primals_117 = add_45 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_118, add_48);  primals_118 = add_48 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_119, add_50);  primals_119 = add_50 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_120, add_51);  primals_120 = add_51 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_121, add_54);  primals_121 = add_54 = None
    copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_122, add_56);  primals_122 = add_56 = None
    copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_123, add_57);  primals_123 = add_57 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_124, add_60);  primals_124 = add_60 = None
    copy__31: "f32[64]" = torch.ops.aten.copy_.default(primals_125, add_62);  primals_125 = add_62 = None
    copy__32: "f32[64]" = torch.ops.aten.copy_.default(primals_126, add_63);  primals_126 = add_63 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_127, add_66);  primals_127 = add_66 = None
    copy__34: "f32[64]" = torch.ops.aten.copy_.default(primals_128, add_68);  primals_128 = add_68 = None
    copy__35: "f32[64]" = torch.ops.aten.copy_.default(primals_129, add_69);  primals_129 = add_69 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_130, add_72);  primals_130 = add_72 = None
    copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_131, add_74);  primals_131 = add_74 = None
    copy__38: "f32[128]" = torch.ops.aten.copy_.default(primals_132, add_75);  primals_132 = add_75 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_133, add_78);  primals_133 = add_78 = None
    copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_134, add_80);  primals_134 = add_80 = None
    copy__41: "f32[128]" = torch.ops.aten.copy_.default(primals_135, add_81);  primals_135 = add_81 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_136, add_84);  primals_136 = add_84 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_137, add_86);  primals_137 = add_86 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_138, add_87);  primals_138 = add_87 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_139, add_90);  primals_139 = add_90 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_140, add_92);  primals_140 = add_92 = None
    copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_141, add_93);  primals_141 = add_93 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_142, add_96);  primals_142 = add_96 = None
    copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_143, add_98);  primals_143 = add_98 = None
    copy__50: "f32[128]" = torch.ops.aten.copy_.default(primals_144, add_99);  primals_144 = add_99 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_145, add_102);  primals_145 = add_102 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_146, add_104);  primals_146 = add_104 = None
    copy__53: "f32[128]" = torch.ops.aten.copy_.default(primals_147, add_105);  primals_147 = add_105 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_148, add_108);  primals_148 = add_108 = None
    copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_149, add_110);  primals_149 = add_110 = None
    copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_150, add_111);  primals_150 = add_111 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_114);  primals_151 = add_114 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_152, add_116);  primals_152 = add_116 = None
    copy__59: "f32[128]" = torch.ops.aten.copy_.default(primals_153, add_117);  primals_153 = add_117 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_154, add_120);  primals_154 = add_120 = None
    copy__61: "f32[128]" = torch.ops.aten.copy_.default(primals_155, add_122);  primals_155 = add_122 = None
    copy__62: "f32[128]" = torch.ops.aten.copy_.default(primals_156, add_123);  primals_156 = add_123 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_157, add_126);  primals_157 = add_126 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_158, add_128);  primals_158 = add_128 = None
    copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_159, add_129);  primals_159 = add_129 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_160, add_132);  primals_160 = add_132 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_161, add_134);  primals_161 = add_134 = None
    copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_162, add_135);  primals_162 = add_135 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_163, add_138);  primals_163 = add_138 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_164, add_140);  primals_164 = add_140 = None
    copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_165, add_141);  primals_165 = add_141 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_166, add_145);  primals_166 = add_145 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_167, add_147);  primals_167 = add_147 = None
    copy__74: "f32[256]" = torch.ops.aten.copy_.default(primals_168, add_148);  primals_168 = add_148 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_169, add_151);  primals_169 = add_151 = None
    copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_170, add_153);  primals_170 = add_153 = None
    copy__77: "f32[256]" = torch.ops.aten.copy_.default(primals_171, add_154);  primals_171 = add_154 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_172, add_158);  primals_172 = add_158 = None
    copy__79: "f32[256]" = torch.ops.aten.copy_.default(primals_173, add_160);  primals_173 = add_160 = None
    copy__80: "f32[256]" = torch.ops.aten.copy_.default(primals_174, add_161);  primals_174 = add_161 = None
    return pytree.tree_unflatten([addmm, mul_495, sum_56, mul_485, sum_54, mul_475, sum_52, mul_465, sum_50, mul_455, sum_48, mul_445, sum_46, mul_435, sum_44, mul_425, sum_42, mul_415, sum_40, mul_405, sum_38, mul_395, sum_36, mul_385, sum_34, mul_375, sum_32, mul_365, sum_30, mul_355, sum_28, mul_345, sum_26, mul_335, sum_24, mul_325, sum_22, mul_315, sum_20, mul_305, sum_18, mul_295, sum_16, mul_285, sum_14, mul_275, sum_12, mul_265, sum_10, mul_252, sum_7, mul_242, sum_5, mul_229, sum_2, permute_4, view_2, getitem_148, getitem_145, getitem_142, getitem_139, getitem_136, getitem_133, getitem_130, getitem_127, getitem_124, getitem_121, getitem_118, getitem_115, getitem_112, getitem_109, getitem_106, getitem_103, getitem_100, getitem_97, getitem_94, getitem_91, getitem_88, getitem_85, getitem_82, getitem_79, getitem_76, getitem_77, getitem_73, getitem_74, getitem_70, getitem_67, getitem_64, getitem_65, getitem_61, getitem_62, getitem_58, getitem_55, getitem_56, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    