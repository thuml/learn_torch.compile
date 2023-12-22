from __future__ import annotations



def forward(self, primals_1: "f32[8]", primals_2: "f32[8]", primals_3: "f32[8]", primals_4: "f32[8]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[16]", primals_8: "f32[16]", primals_9: "f32[32]", primals_10: "f32[32]", primals_11: "f32[32]", primals_12: "f32[32]", primals_13: "f32[32]", primals_14: "f32[32]", primals_15: "f32[32]", primals_16: "f32[32]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[64]", primals_21: "f32[64]", primals_22: "f32[64]", primals_23: "f32[64]", primals_24: "f32[64]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[128]", primals_34: "f32[128]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[256]", primals_50: "f32[256]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[1000, 1280]", primals_56: "f32[1000]", primals_57: "f32[8, 3, 3, 3]", primals_58: "f32[8, 1, 3, 3]", primals_59: "f32[16, 8, 1, 1]", primals_60: "f32[16, 1, 3, 3]", primals_61: "f32[32, 16, 1, 1]", primals_62: "f32[32, 1, 3, 3]", primals_63: "f32[32, 32, 1, 1]", primals_64: "f32[32, 1, 3, 3]", primals_65: "f32[64, 32, 1, 1]", primals_66: "f32[64, 1, 3, 3]", primals_67: "f32[64, 64, 1, 1]", primals_68: "f32[64, 1, 3, 3]", primals_69: "f32[128, 64, 1, 1]", primals_70: "f32[128, 1, 5, 5]", primals_71: "f32[128, 128, 1, 1]", primals_72: "f32[128, 1, 5, 5]", primals_73: "f32[128, 128, 1, 1]", primals_74: "f32[128, 1, 5, 5]", primals_75: "f32[128, 128, 1, 1]", primals_76: "f32[128, 1, 5, 5]", primals_77: "f32[128, 128, 1, 1]", primals_78: "f32[128, 1, 5, 5]", primals_79: "f32[128, 128, 1, 1]", primals_80: "f32[128, 1, 5, 5]", primals_81: "f32[32, 128, 1, 1]", primals_82: "f32[32]", primals_83: "f32[128, 32, 1, 1]", primals_84: "f32[128]", primals_85: "f32[256, 128, 1, 1]", primals_86: "f32[256, 1, 5, 5]", primals_87: "f32[64, 256, 1, 1]", primals_88: "f32[64]", primals_89: "f32[256, 64, 1, 1]", primals_90: "f32[256]", primals_91: "f32[256, 256, 1, 1]", primals_92: "f32[1280, 256, 1, 1]", primals_93: "f32[1280]", primals_94: "i64[]", primals_95: "f32[8]", primals_96: "f32[8]", primals_97: "i64[]", primals_98: "f32[8]", primals_99: "f32[8]", primals_100: "i64[]", primals_101: "f32[16]", primals_102: "f32[16]", primals_103: "i64[]", primals_104: "f32[16]", primals_105: "f32[16]", primals_106: "i64[]", primals_107: "f32[32]", primals_108: "f32[32]", primals_109: "i64[]", primals_110: "f32[32]", primals_111: "f32[32]", primals_112: "i64[]", primals_113: "f32[32]", primals_114: "f32[32]", primals_115: "i64[]", primals_116: "f32[32]", primals_117: "f32[32]", primals_118: "i64[]", primals_119: "f32[64]", primals_120: "f32[64]", primals_121: "i64[]", primals_122: "f32[64]", primals_123: "f32[64]", primals_124: "i64[]", primals_125: "f32[64]", primals_126: "f32[64]", primals_127: "i64[]", primals_128: "f32[64]", primals_129: "f32[64]", primals_130: "i64[]", primals_131: "f32[128]", primals_132: "f32[128]", primals_133: "i64[]", primals_134: "f32[128]", primals_135: "f32[128]", primals_136: "i64[]", primals_137: "f32[128]", primals_138: "f32[128]", primals_139: "i64[]", primals_140: "f32[128]", primals_141: "f32[128]", primals_142: "i64[]", primals_143: "f32[128]", primals_144: "f32[128]", primals_145: "i64[]", primals_146: "f32[128]", primals_147: "f32[128]", primals_148: "i64[]", primals_149: "f32[128]", primals_150: "f32[128]", primals_151: "i64[]", primals_152: "f32[128]", primals_153: "f32[128]", primals_154: "i64[]", primals_155: "f32[128]", primals_156: "f32[128]", primals_157: "i64[]", primals_158: "f32[128]", primals_159: "f32[128]", primals_160: "i64[]", primals_161: "f32[128]", primals_162: "f32[128]", primals_163: "i64[]", primals_164: "f32[128]", primals_165: "f32[128]", primals_166: "i64[]", primals_167: "f32[256]", primals_168: "f32[256]", primals_169: "i64[]", primals_170: "f32[256]", primals_171: "f32[256]", primals_172: "i64[]", primals_173: "f32[256]", primals_174: "f32[256]", primals_175: "f32[8, 3, 224, 224]"):
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
    add_164: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_31, 3)
    clamp_min_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_164, 0);  add_164 = None
    clamp_max_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_218: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_31, clamp_max_29);  clamp_max_29 = None
    div_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_218, 6);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_1: "f32[8, 1280]" = torch.ops.aten.view.default(div_29, [8, 1280]);  div_29 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_56, view_1, permute);  primals_56 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_108: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_109: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
    unsqueeze_110: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt: "b8[8, 256, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_29, -3.0)
    lt_2: "b8[8, 256, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_29, 3.0);  convolution_29 = None
    bitwise_and: "b8[8, 256, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_121: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
    unsqueeze_122: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_132: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_133: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
    unsqueeze_134: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_1: "b8[8, 128, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_25, -3.0)
    lt_5: "b8[8, 128, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_25, 3.0);  convolution_25 = None
    bitwise_and_1: "b8[8, 128, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_5);  gt_1 = lt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_145: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
    unsqueeze_146: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_156: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_157: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_169: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_180: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_181: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_192: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_193: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_216: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_217: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_228: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_229: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_253: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_264: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_265: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_276: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_277: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_288: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_289: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_300: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_301: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_312: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_313: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_324: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_325: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_336: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_337: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_348: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_349: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_360: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_361: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_372: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_373: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_384: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_385: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_397: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_409: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_421: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    
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
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_89, primals_91, primals_92, primals_175, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, clone_2, div_2, convolution_3, squeeze_10, clone_3, div_3, convolution_4, squeeze_13, clone_4, div_4, convolution_5, squeeze_16, clone_5, div_5, convolution_6, squeeze_19, clone_6, div_6, convolution_7, squeeze_22, clone_7, div_7, convolution_8, squeeze_25, clone_8, div_8, convolution_9, squeeze_28, clone_9, div_9, convolution_10, squeeze_31, clone_10, div_10, convolution_11, squeeze_34, clone_11, div_11, convolution_12, squeeze_37, clone_12, div_12, convolution_13, squeeze_40, clone_13, div_13, convolution_14, squeeze_43, clone_14, div_14, convolution_15, squeeze_46, clone_15, div_15, convolution_16, squeeze_49, clone_16, div_16, convolution_17, squeeze_52, clone_17, div_17, convolution_18, squeeze_55, clone_18, div_18, convolution_19, squeeze_58, clone_19, div_19, convolution_20, squeeze_61, clone_20, div_20, convolution_21, squeeze_64, clone_21, div_21, convolution_22, squeeze_67, clone_22, div_22, convolution_23, squeeze_70, clone_23, div_23, mean, relu, div_24, mul_192, convolution_26, squeeze_73, clone_24, div_25, convolution_27, squeeze_76, clone_25, div_26, mean_1, relu_1, div_27, mul_209, convolution_30, squeeze_79, clone_26, mean_2, convolution_31, view_1, permute_1, unsqueeze_110, bitwise_and, unsqueeze_122, unsqueeze_134, bitwise_and_1, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422]
    