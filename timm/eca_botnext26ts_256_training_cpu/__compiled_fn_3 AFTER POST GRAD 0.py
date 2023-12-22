from __future__ import annotations



def forward(self, primals_1: "f32[24]", primals_2: "f32[24]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[64]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[256]", primals_14: "f32[256]", primals_15: "f32[64]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[256]", primals_20: "f32[256]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[256]", primals_38: "f32[256]", primals_39: "f32[1024]", primals_40: "f32[1024]", primals_41: "f32[1024]", primals_42: "f32[1024]", primals_43: "f32[256]", primals_44: "f32[256]", primals_45: "f32[31, 16]", primals_46: "f32[31, 16]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[1024]", primals_50: "f32[1024]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[31, 16]", primals_54: "f32[31, 16]", primals_55: "f32[512]", primals_56: "f32[512]", primals_57: "f32[2048]", primals_58: "f32[2048]", primals_59: "f32[2048]", primals_60: "f32[2048]", primals_61: "f32[512]", primals_62: "f32[512]", primals_63: "f32[15, 16]", primals_64: "f32[15, 16]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[2048]", primals_68: "f32[2048]", primals_69: "f32[24, 3, 3, 3]", primals_70: "f32[32, 24, 3, 3]", primals_71: "f32[64, 32, 3, 3]", primals_72: "f32[64, 64, 1, 1]", primals_73: "f32[64, 16, 3, 3]", primals_74: "f32[1, 1, 3]", primals_75: "f32[256, 64, 1, 1]", primals_76: "f32[256, 64, 1, 1]", primals_77: "f32[64, 256, 1, 1]", primals_78: "f32[64, 16, 3, 3]", primals_79: "f32[1, 1, 3]", primals_80: "f32[256, 64, 1, 1]", primals_81: "f32[128, 256, 1, 1]", primals_82: "f32[128, 16, 3, 3]", primals_83: "f32[1, 1, 5]", primals_84: "f32[512, 128, 1, 1]", primals_85: "f32[512, 256, 1, 1]", primals_86: "f32[128, 512, 1, 1]", primals_87: "f32[128, 16, 3, 3]", primals_88: "f32[1, 1, 5]", primals_89: "f32[512, 128, 1, 1]", primals_90: "f32[256, 512, 1, 1]", primals_91: "f32[256, 16, 3, 3]", primals_92: "f32[1, 1, 5]", primals_93: "f32[1024, 256, 1, 1]", primals_94: "f32[1024, 512, 1, 1]", primals_95: "f32[256, 1024, 1, 1]", primals_96: "f32[384, 256, 1, 1]", primals_97: "f32[1024, 256, 1, 1]", primals_98: "f32[512, 1024, 1, 1]", primals_99: "f32[640, 512, 1, 1]", primals_100: "f32[2048, 512, 1, 1]", primals_101: "f32[2048, 1024, 1, 1]", primals_102: "f32[512, 2048, 1, 1]", primals_103: "f32[640, 512, 1, 1]", primals_104: "f32[2048, 512, 1, 1]", primals_105: "f32[1000, 2048]", primals_106: "f32[1000]", primals_107: "i64[]", primals_108: "f32[24]", primals_109: "f32[24]", primals_110: "i64[]", primals_111: "f32[32]", primals_112: "f32[32]", primals_113: "i64[]", primals_114: "f32[64]", primals_115: "f32[64]", primals_116: "i64[]", primals_117: "f32[64]", primals_118: "f32[64]", primals_119: "i64[]", primals_120: "f32[64]", primals_121: "f32[64]", primals_122: "i64[]", primals_123: "f32[256]", primals_124: "f32[256]", primals_125: "i64[]", primals_126: "f32[256]", primals_127: "f32[256]", primals_128: "i64[]", primals_129: "f32[64]", primals_130: "f32[64]", primals_131: "i64[]", primals_132: "f32[64]", primals_133: "f32[64]", primals_134: "i64[]", primals_135: "f32[256]", primals_136: "f32[256]", primals_137: "i64[]", primals_138: "f32[128]", primals_139: "f32[128]", primals_140: "i64[]", primals_141: "f32[128]", primals_142: "f32[128]", primals_143: "i64[]", primals_144: "f32[512]", primals_145: "f32[512]", primals_146: "i64[]", primals_147: "f32[512]", primals_148: "f32[512]", primals_149: "i64[]", primals_150: "f32[128]", primals_151: "f32[128]", primals_152: "i64[]", primals_153: "f32[128]", primals_154: "f32[128]", primals_155: "i64[]", primals_156: "f32[512]", primals_157: "f32[512]", primals_158: "i64[]", primals_159: "f32[256]", primals_160: "f32[256]", primals_161: "i64[]", primals_162: "f32[256]", primals_163: "f32[256]", primals_164: "i64[]", primals_165: "f32[1024]", primals_166: "f32[1024]", primals_167: "i64[]", primals_168: "f32[1024]", primals_169: "f32[1024]", primals_170: "i64[]", primals_171: "f32[256]", primals_172: "f32[256]", primals_173: "i64[]", primals_174: "f32[256]", primals_175: "f32[256]", primals_176: "i64[]", primals_177: "f32[1024]", primals_178: "f32[1024]", primals_179: "i64[]", primals_180: "f32[512]", primals_181: "f32[512]", primals_182: "i64[]", primals_183: "f32[512]", primals_184: "f32[512]", primals_185: "i64[]", primals_186: "f32[2048]", primals_187: "f32[2048]", primals_188: "i64[]", primals_189: "f32[2048]", primals_190: "f32[2048]", primals_191: "i64[]", primals_192: "f32[512]", primals_193: "f32[512]", primals_194: "i64[]", primals_195: "f32[512]", primals_196: "f32[512]", primals_197: "i64[]", primals_198: "f32[2048]", primals_199: "f32[2048]", primals_200: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(primals_200, primals_69, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_107, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 24, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 24, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[24]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[24]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[24]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
    add_3: "f32[24]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_7, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_110, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_15, primals_71, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_113, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(mul_23, [3, 3], [2, 2], [1, 1])
    getitem_6: "f32[8, 64, 64, 64]" = max_pool2d_with_indices[0]
    getitem_7: "i64[8, 64, 64, 64]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_116, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_24: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_19)
    mul_31: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_3);  sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_31, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_119, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_32: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000030518509476);  squeeze_14 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_38: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_17);  mul_32 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_19);  mul_38 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_39, [2, 3])
    view: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(mean, [8, 1, -1]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_5: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view, primals_74, None, [1], [1], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_1: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_5, [8, -1, 1, 1]);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
    mul_40: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, expand);  mul_39 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_40, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_122, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_41: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(mul_44, 0.1);  mul_44 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_47: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_21);  mul_41 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_23);  mul_47 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_125, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_48: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_54: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_25);  mul_48 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_27);  mul_54 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_35: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_29, add_34);  add_29 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_35)
    mul_55: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_55, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_128, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
    add_39: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_62: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_29);  mul_56 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_31);  mul_62 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_40)
    mul_63: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, sigmoid_7);  sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_63, primals_78, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_131, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_64: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
    add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_70: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_33);  mul_64 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_35);  mul_70 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    mul_71: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_1: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_71, [2, 3])
    view_2: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(mean_1, [8, 1, -1]);  mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_10: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view_2, primals_79, None, [1], [1], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_3: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_9, [8, -1, 1, 1]);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_1: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_71, expand_1);  mul_71 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_72, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_134, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 256, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 256, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_21)
    mul_73: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
    add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_76: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_77: "f32[256]" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
    add_49: "f32[256]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
    unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_79: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_37);  mul_73 = unsqueeze_37 = None
    unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_39);  mul_79 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_51: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_50, mul_55);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_51)
    mul_80: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_80, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_137, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_23)
    mul_81: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_83: "f32[128]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
    add_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_87: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_41);  mul_81 = unsqueeze_41 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_43);  mul_87 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_11: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
    mul_88: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_88, primals_82, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_140, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1, 1]" = var_mean_11[0]
    getitem_25: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_25)
    mul_89: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001220852154804);  squeeze_35 = None
    mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(mul_92, 0.1);  mul_92 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_95: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_45);  mul_89 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_47);  mul_95 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    mul_96: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_2: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_96, [2, 3])
    view_4: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mean_2, [8, 1, -1]);  mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_14: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_4, primals_83, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_5: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_13, [8, -1, 1, 1]);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_2: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
    mul_97: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_96, expand_2);  mul_96 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_97, primals_84, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_143, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1, 1]" = var_mean_12[0]
    getitem_27: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_27)
    mul_98: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001220852154804);  squeeze_38 = None
    mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
    add_65: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_104: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_49);  mul_98 = unsqueeze_49 = None
    unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_51);  mul_104 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_80, primals_85, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_146, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1, 1]" = var_mean_13[0]
    getitem_29: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_29)
    mul_105: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_106: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_107: "f32[512]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
    add_69: "f32[512]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_108: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
    mul_109: "f32[512]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[512]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
    add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_111: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_53);  mul_105 = unsqueeze_53 = None
    unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_55);  mul_111 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_72: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_66, add_71);  add_66 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_14: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_72)
    mul_112: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_112, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_149, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_31: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_31)
    mul_113: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
    add_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_119: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_57);  mul_113 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_59);  mul_119 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_77)
    mul_120: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_77, sigmoid_15);  sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_120, primals_87, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_152, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_33: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_33)
    mul_121: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_123: "f32[128]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_125: "f32[128]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[128]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
    add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_127: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_61);  mul_121 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_63);  mul_127 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    mul_128: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_3: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_128, [2, 3])
    view_6: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mean_3, [8, 1, -1]);  mean_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_19: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_6, primals_88, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_7: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_17, [8, -1, 1, 1]);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_3: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
    mul_129: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, expand_3);  mul_128 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_129, primals_89, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_155, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1, 1]" = var_mean_16[0]
    getitem_35: "f32[1, 512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_16: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_35)
    mul_130: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_49: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_131: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_85: "f32[512]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_50: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_133: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_134: "f32[512]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[512]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
    add_86: "f32[512]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    unsqueeze_64: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_136: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_65);  mul_130 = unsqueeze_65 = None
    unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_67);  mul_136 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_88: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_87, mul_112);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_18: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_88)
    mul_137: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_137, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 256, 1, 1]" = var_mean_17[0]
    getitem_37: "f32[1, 256, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_17: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_37)
    mul_138: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_52: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_139: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_140: "f32[256]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_91: "f32[256]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_53: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_141: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_142: "f32[256]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[256]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
    add_92: "f32[256]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_144: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_69);  mul_138 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_71);  mul_144 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_19: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    mul_145: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_19);  sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_145, primals_91, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_161, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 256, 1, 1]" = var_mean_18[0]
    getitem_39: "f32[1, 256, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_18: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_39)
    mul_146: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_55: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_147: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_96: "f32[256]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    squeeze_56: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_149: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0004885197850513);  squeeze_56 = None
    mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
    mul_151: "f32[256]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
    add_97: "f32[256]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_152: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_73);  mul_146 = unsqueeze_73 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_75);  mul_152 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_98)
    mul_153: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_98, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_4: "f32[8, 256]" = torch.ops.aten.mean.dim(mul_153, [2, 3])
    view_8: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mean_4, [8, 1, -1]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_23: "f32[8, 1, 256]" = torch.ops.aten.convolution.default(view_8, primals_92, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_21: "f32[8, 1, 256]" = torch.ops.aten.sigmoid.default(convolution_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_9: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_21, [8, -1, 1, 1]);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_4: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
    mul_154: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_153, expand_4);  mul_153 = expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_154, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1, 1]" = var_mean_19[0]
    getitem_41: "f32[1, 1024, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_19: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_41)
    mul_155: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_58: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_156: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_157: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_101: "f32[1024]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_59: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_158: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0004885197850513);  squeeze_59 = None
    mul_159: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_102: "f32[1024]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_161: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_77);  mul_155 = unsqueeze_77 = None
    unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_79);  mul_161 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_137, primals_94, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1, 1]" = var_mean_20[0]
    getitem_43: "f32[1, 1024, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_20: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_43)
    mul_162: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_61: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_163: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_164: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_106: "f32[1024]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    squeeze_62: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_165: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
    mul_166: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_165, 0.1);  mul_165 = None
    mul_167: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
    add_107: "f32[1024]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    unsqueeze_80: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_168: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_81);  mul_162 = unsqueeze_81 = None
    unsqueeze_82: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_83);  mul_168 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_109: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_103, add_108);  add_103 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_22: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_109)
    mul_169: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_109, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_169, primals_95, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_170, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_21[0]
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_21: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_45)
    mul_170: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_64: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_171: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_112: "f32[256]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    squeeze_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(mul_173, 0.1);  mul_173 = None
    mul_175: "f32[256]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_176: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_85);  mul_170 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_87);  mul_176 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_114)
    mul_177: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_114, sigmoid_23);  sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_27: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_177, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(convolution_27, [64, 64, 256], 1);  convolution_27 = None
    getitem_46: "f32[8, 64, 16, 16]" = split_with_sizes[0]
    getitem_47: "f32[8, 64, 16, 16]" = split_with_sizes[1]
    getitem_48: "f32[8, 256, 16, 16]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_19: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_46, memory_format = torch.contiguous_format);  getitem_46 = None
    view_10: "f32[32, 16, 256]" = torch.ops.aten.reshape.default(clone_19, [32, 16, 256]);  clone_19 = None
    permute: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_20: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_47, memory_format = torch.contiguous_format);  getitem_47 = None
    view_11: "f32[32, 16, 256]" = torch.ops.aten.reshape.default(clone_20, [32, 16, 256]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_21: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_48, memory_format = torch.contiguous_format);  getitem_48 = None
    view_12: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_21, [32, 64, 256]);  clone_21 = None
    permute_1: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_5: "f32[32, 256, 16]" = torch.ops.aten.expand.default(permute, [32, 256, 16])
    expand_6: "f32[32, 16, 256]" = torch.ops.aten.expand.default(view_11, [32, 16, 256]);  view_11 = None
    bmm: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_5, expand_6)
    mul_178: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm, 0.25);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_16: "f32[32, 16, 16, 16]" = torch.ops.aten.reshape.default(permute, [32, 16, 16, 16]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_2: "f32[16, 31]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_22: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format)
    view_17: "f32[8192, 16]" = torch.ops.aten.reshape.default(clone_22, [8192, 16]);  clone_22 = None
    mm: "f32[8192, 31]" = torch.ops.aten.mm.default(view_17, permute_2)
    view_18: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm, [32, 16, 16, 31]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_19: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_18, [-1, 16, 31]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_19, [0, 1], 0.0);  view_19 = None
    view_20: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd, [512, 512]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_1: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_20, [0, 15], 0.0);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_21: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_1, [-1, 17, 31]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_2: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_21, 1, 0, 16);  view_21 = None
    slice_3: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_2, 2, 15, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_22: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_3, [32, 16, 1, 16, 16]);  slice_3 = None
    expand_7: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_22, [-1, -1, 16, -1, -1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_3: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_7, [0, 1, 3, 2, 4]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_4: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_5: "f32[16, 31]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    clone_23: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_23: "f32[8192, 16]" = torch.ops.aten.reshape.default(clone_23, [8192, 16]);  clone_23 = None
    mm_1: "f32[8192, 31]" = torch.ops.aten.mm.default(view_23, permute_5)
    view_24: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_1, [32, 16, 16, 31]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_25: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_24, [-1, 16, 31]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_2: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_25, [0, 1], 0.0);  view_25 = None
    view_26: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_2, [512, 512]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_3: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_26, [0, 15], 0.0);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_27: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_3, [-1, 17, 31]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_5: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_27, 1, 0, 16);  view_27 = None
    slice_6: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_5, 2, 15, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_28: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_6, [32, 16, 1, 16, 16]);  slice_6 = None
    expand_8: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_28, [-1, -1, 16, -1, -1]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_6: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_8, [0, 3, 1, 4, 2]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_115: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_6, permute_3);  permute_6 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_24: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format);  add_115 = None
    view_29: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_24, [32, 256, 256]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_116: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_178, view_29);  mul_178 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
    sub_22: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_116, amax);  add_116 = amax = None
    exp: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_1: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_5: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_9: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div, [32, 256, 256]);  div = None
    expand_10: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_1, [32, 256, 64]);  permute_1 = None
    bmm_1: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(expand_9, expand_10)
    permute_7: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1])
    clone_25: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_33: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_25, [8, 256, 16, 16]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_49: "f32[1, 256, 1, 1]" = var_mean_22[0]
    getitem_50: "f32[1, 256, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05)
    rsqrt_22: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_33, getitem_50);  view_33 = None
    mul_179: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
    squeeze_66: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    squeeze_67: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_180: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_181: "f32[256]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_119: "f32[256]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    squeeze_68: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    mul_182: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_183: "f32[256]" = torch.ops.aten.mul.Tensor(mul_182, 0.1);  mul_182 = None
    mul_184: "f32[256]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_120: "f32[256]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_185: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_89);  mul_179 = unsqueeze_89 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_91);  mul_185 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_121)
    mul_186: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_121, sigmoid_24);  sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_186, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_176, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_51: "f32[1, 1024, 1, 1]" = var_mean_23[0]
    getitem_52: "f32[1, 1024, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-05)
    rsqrt_23: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_52)
    mul_187: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
    squeeze_69: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    squeeze_70: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_188: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_189: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_124: "f32[1024]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    squeeze_71: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_190, 0.1);  mul_190 = None
    mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_125: "f32[1024]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    unsqueeze_92: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_93: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_193: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_93);  mul_187 = unsqueeze_93 = None
    unsqueeze_94: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_95: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_95);  mul_193 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_127: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_126, mul_169);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    mul_194: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_194, primals_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_53: "f32[1, 512, 1, 1]" = var_mean_24[0]
    getitem_54: "f32[1, 512, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05)
    rsqrt_24: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_54)
    mul_195: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    squeeze_72: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    squeeze_73: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_196: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_197: "f32[512]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_130: "f32[512]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    squeeze_74: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    mul_198: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_199: "f32[512]" = torch.ops.aten.mul.Tensor(mul_198, 0.1);  mul_198 = None
    mul_200: "f32[512]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_131: "f32[512]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_201: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_97);  mul_195 = unsqueeze_97 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_99);  mul_201 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_26: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    mul_202: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_30: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_202, primals_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(convolution_30, [64, 64, 512], 1);  convolution_30 = None
    getitem_55: "f32[8, 64, 16, 16]" = split_with_sizes_1[0]
    getitem_56: "f32[8, 64, 16, 16]" = split_with_sizes_1[1]
    getitem_57: "f32[8, 512, 16, 16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_29: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_55, memory_format = torch.contiguous_format);  getitem_55 = None
    view_34: "f32[32, 16, 256]" = torch.ops.aten.reshape.default(clone_29, [32, 16, 256]);  clone_29 = None
    permute_8: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_30: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_56, memory_format = torch.contiguous_format);  getitem_56 = None
    view_35: "f32[32, 16, 256]" = torch.ops.aten.reshape.default(clone_30, [32, 16, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_31: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    view_36: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_31, [32, 128, 256]);  clone_31 = None
    permute_9: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_11: "f32[32, 256, 16]" = torch.ops.aten.expand.default(permute_8, [32, 256, 16])
    expand_12: "f32[32, 16, 256]" = torch.ops.aten.expand.default(view_35, [32, 16, 256]);  view_35 = None
    bmm_2: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_11, expand_12)
    mul_203: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_2, 0.25);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_40: "f32[32, 16, 16, 16]" = torch.ops.aten.reshape.default(permute_8, [32, 16, 16, 16]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_10: "f32[16, 31]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    clone_32: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(view_40, memory_format = torch.contiguous_format)
    view_41: "f32[8192, 16]" = torch.ops.aten.reshape.default(clone_32, [8192, 16]);  clone_32 = None
    mm_2: "f32[8192, 31]" = torch.ops.aten.mm.default(view_41, permute_10)
    view_42: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_2, [32, 16, 16, 31]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_43: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_42, [-1, 16, 31]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_4: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_43, [0, 1], 0.0);  view_43 = None
    view_44: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_4, [512, 512]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_5: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_44, [0, 15], 0.0);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_45: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_5, [-1, 17, 31]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_8: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_45, 1, 0, 16);  view_45 = None
    slice_9: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_8, 2, 15, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_46: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_9, [32, 16, 1, 16, 16]);  slice_9 = None
    expand_13: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_46, [-1, -1, 16, -1, -1]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_11: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_13, [0, 1, 3, 2, 4]);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_12: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_13: "f32[16, 31]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    clone_33: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_47: "f32[8192, 16]" = torch.ops.aten.reshape.default(clone_33, [8192, 16]);  clone_33 = None
    mm_3: "f32[8192, 31]" = torch.ops.aten.mm.default(view_47, permute_13)
    view_48: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_3, [32, 16, 16, 31]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_49: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_48, [-1, 16, 31]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_6: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_49, [0, 1], 0.0);  view_49 = None
    view_50: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_6, [512, 512]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_7: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_50, [0, 15], 0.0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_51: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_7, [-1, 17, 31]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_11: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_51, 1, 0, 16);  view_51 = None
    slice_12: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_11, 2, 15, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_52: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_12, [32, 16, 1, 16, 16]);  slice_12 = None
    expand_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_52, [-1, -1, 16, -1, -1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_14, [0, 3, 1, 4, 2]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_133: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_14, permute_11);  permute_14 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_34: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format);  add_133 = None
    view_53: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_34, [32, 256, 256]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_134: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_203, view_53);  mul_203 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_134, [-1], True)
    sub_26: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_134, amax_1);  add_134 = amax_1 = None
    exp_1: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_6: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_15: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_1, [32, 256, 256]);  div_1 = None
    expand_16: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_9, [32, 256, 128]);  permute_9 = None
    bmm_3: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(expand_15, expand_16)
    permute_15: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_3, [0, 2, 1]);  bmm_3 = None
    clone_35: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_57: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(clone_35, [8, 512, 16, 16]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_57, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_182, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(avg_pool2d, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1, 1]" = var_mean_25[0]
    getitem_59: "f32[1, 512, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_136: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_25: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_59)
    mul_204: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_25);  sub_27 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_137: "f32[512]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0019569471624266);  squeeze_77 = None
    mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(mul_207, 0.1);  mul_207 = None
    mul_209: "f32[512]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_138: "f32[512]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_210: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_101);  mul_204 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_139: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_103);  mul_210 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_139)
    mul_211: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_139, sigmoid_27);  sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_211, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 2048, 1, 1]" = var_mean_26[0]
    getitem_61: "f32[1, 2048, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_141: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_26: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_61)
    mul_212: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_26);  sub_28 = None
    squeeze_78: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_79: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_213: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_214: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_142: "f32[2048]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_80: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_215: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0019569471624266);  squeeze_80 = None
    mul_216: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_143: "f32[2048]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_104: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_105: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_218: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_105);  mul_212 = unsqueeze_105 = None
    unsqueeze_106: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_107: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_144: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_107);  mul_218 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_194, primals_101, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_188, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 2048, 1, 1]" = var_mean_27[0]
    getitem_63: "f32[1, 2048, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_146: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_27: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_29: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_63)
    mul_219: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_27);  sub_29 = None
    squeeze_81: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_82: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_220: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_221: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_147: "f32[2048]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    squeeze_83: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0019569471624266);  squeeze_83 = None
    mul_223: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_222, 0.1);  mul_222 = None
    mul_224: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_148: "f32[2048]" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    unsqueeze_108: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_109: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_225: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_109);  mul_219 = unsqueeze_109 = None
    unsqueeze_110: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_111: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_149: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_225, unsqueeze_111);  mul_225 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_150: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_144, add_149);  add_144 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_150)
    mul_226: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_28);  sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_226, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512, 1, 1]" = var_mean_28[0]
    getitem_65: "f32[1, 512, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_28: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_65)
    mul_227: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
    squeeze_84: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_85: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_153: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    squeeze_86: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0019569471624266);  squeeze_86 = None
    mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(mul_230, 0.1);  mul_230 = None
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_233: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_113);  mul_227 = unsqueeze_113 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_115);  mul_233 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_155)
    mul_234: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_234, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(convolution_34, [64, 64, 512], 1);  convolution_34 = None
    getitem_66: "f32[8, 64, 8, 8]" = split_with_sizes_2[0]
    getitem_67: "f32[8, 64, 8, 8]" = split_with_sizes_2[1]
    getitem_68: "f32[8, 512, 8, 8]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_39: "f32[8, 64, 8, 8]" = torch.ops.aten.clone.default(getitem_66, memory_format = torch.contiguous_format);  getitem_66 = None
    view_58: "f32[32, 16, 64]" = torch.ops.aten.reshape.default(clone_39, [32, 16, 64]);  clone_39 = None
    permute_16: "f32[32, 64, 16]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_40: "f32[8, 64, 8, 8]" = torch.ops.aten.clone.default(getitem_67, memory_format = torch.contiguous_format);  getitem_67 = None
    view_59: "f32[32, 16, 64]" = torch.ops.aten.reshape.default(clone_40, [32, 16, 64]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_41: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_68, memory_format = torch.contiguous_format);  getitem_68 = None
    view_60: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_41, [32, 128, 64]);  clone_41 = None
    permute_17: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_17: "f32[32, 64, 16]" = torch.ops.aten.expand.default(permute_16, [32, 64, 16])
    expand_18: "f32[32, 16, 64]" = torch.ops.aten.expand.default(view_59, [32, 16, 64]);  view_59 = None
    bmm_4: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(expand_17, expand_18)
    mul_235: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_4, 0.25);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_64: "f32[32, 8, 8, 16]" = torch.ops.aten.reshape.default(permute_16, [32, 8, 8, 16]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[16, 15]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    clone_42: "f32[32, 8, 8, 16]" = torch.ops.aten.clone.default(view_64, memory_format = torch.contiguous_format)
    view_65: "f32[2048, 16]" = torch.ops.aten.reshape.default(clone_42, [2048, 16]);  clone_42 = None
    mm_4: "f32[2048, 15]" = torch.ops.aten.mm.default(view_65, permute_18)
    view_66: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_4, [32, 8, 8, 15]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_67: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_66, [-1, 8, 15]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_67, [0, 1], 0.0);  view_67 = None
    view_68: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_8, [256, 128]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_9: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_68, [0, 7], 0.0);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_69: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_9, [-1, 9, 15]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_14: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_69, 1, 0, 8);  view_69 = None
    slice_15: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_14, 2, 7, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_70: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_15, [32, 8, 1, 8, 8]);  slice_15 = None
    expand_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_70, [-1, -1, 8, -1, -1]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_19, [0, 1, 3, 2, 4]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_20: "f32[32, 8, 8, 16]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_21: "f32[16, 15]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    clone_43: "f32[32, 8, 8, 16]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_71: "f32[2048, 16]" = torch.ops.aten.reshape.default(clone_43, [2048, 16]);  clone_43 = None
    mm_5: "f32[2048, 15]" = torch.ops.aten.mm.default(view_71, permute_21)
    view_72: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_5, [32, 8, 8, 15]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_73: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_72, [-1, 8, 15]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_10: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_73, [0, 1], 0.0);  view_73 = None
    view_74: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_10, [256, 128]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_11: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_74, [0, 7], 0.0);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_75: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_11, [-1, 9, 15]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_17: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_75, 1, 0, 8);  view_75 = None
    slice_18: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_17, 2, 7, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_76: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_18, [32, 8, 1, 8, 8]);  slice_18 = None
    expand_20: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_76, [-1, -1, 8, -1, -1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_22: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_20, [0, 3, 1, 4, 2]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_156: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_22, permute_19);  permute_22 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_44: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_156, memory_format = torch.contiguous_format);  add_156 = None
    view_77: "f32[32, 64, 64]" = torch.ops.aten.reshape.default(clone_44, [32, 64, 64]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_157: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_235, view_77);  mul_235 = view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
    sub_31: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_157, amax_2);  add_157 = amax_2 = None
    exp_2: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_3: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_7: "f32[32, 64, 64]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_21: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_2, [32, 64, 64]);  div_2 = None
    expand_22: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_17, [32, 64, 128]);  permute_17 = None
    bmm_5: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(expand_21, expand_22)
    permute_23: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_5, [0, 2, 1])
    clone_45: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_81: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_45, [8, 512, 8, 8]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_69: "f32[1, 512, 1, 1]" = var_mean_29[0]
    getitem_70: "f32[1, 512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_159: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_29: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_32: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_81, getitem_70);  view_81 = None
    mul_236: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_29);  sub_32 = None
    squeeze_87: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    squeeze_88: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_238: "f32[512]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_160: "f32[512]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    squeeze_89: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    mul_239: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
    mul_240: "f32[512]" = torch.ops.aten.mul.Tensor(mul_239, 0.1);  mul_239 = None
    mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
    add_161: "f32[512]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_242: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_117);  mul_236 = unsqueeze_117 = None
    unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_162: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_119);  mul_242 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_162)
    mul_243: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_162, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_243, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_71: "f32[1, 2048, 1, 1]" = var_mean_30[0]
    getitem_72: "f32[1, 2048, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_164: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_30: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_33: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_72)
    mul_244: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_30);  sub_33 = None
    squeeze_90: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    squeeze_91: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_245: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_246: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_165: "f32[2048]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    squeeze_92: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_247: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
    mul_248: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_247, 0.1);  mul_247 = None
    mul_249: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_166: "f32[2048]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    unsqueeze_120: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_121: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_250: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_121);  mul_244 = unsqueeze_121 = None
    unsqueeze_122: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_123: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_167: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_123);  mul_250 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_168: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_167, mul_226);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_31: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_168)
    mul_251: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_168, sigmoid_31);  sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_5: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(mul_251, [-1, -2], True);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_82: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean_5, [8, 2048]);  mean_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_24: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_106, view_82, permute_24);  primals_106 = None
    permute_25: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_32: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_168)
    full_default: "f32[8, 2048, 8, 8]" = torch.ops.aten.full.default([8, 2048, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_34: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_32)
    mul_252: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_168, sub_34);  add_168 = sub_34 = None
    add_169: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Scalar(mul_252, 1);  mul_252 = None
    mul_253: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_32, add_169);  sigmoid_32 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_124: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_125: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
    unsqueeze_126: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_162)
    full_default_1: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_39: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_33)
    mul_264: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_162, sub_39);  add_162 = sub_39 = None
    add_170: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_264, 1);  mul_264 = None
    mul_265: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_33, add_170);  sigmoid_33 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_137: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
    unsqueeze_138: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_32: "f32[32, 64, 64]" = torch.ops.aten.permute.default(expand_21, [0, 2, 1]);  expand_21 = None
    permute_33: "f32[32, 128, 64]" = torch.ops.aten.permute.default(expand_22, [0, 2, 1]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_8: "f32[32, 64, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_37: "f32[15, 16]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_43: "f32[15, 16]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_45: "f32[32, 16, 64]" = torch.ops.aten.permute.default(expand_17, [0, 2, 1]);  expand_17 = None
    permute_46: "f32[32, 64, 16]" = torch.ops.aten.permute.default(expand_18, [0, 2, 1]);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_155)
    sub_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_34)
    mul_279: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_155, sub_45);  add_155 = sub_45 = None
    add_173: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_279, 1);  mul_279 = None
    mul_280: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_34, add_173);  sigmoid_34 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_149: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
    unsqueeze_150: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_150)
    sub_50: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_35);  full_default = None
    mul_291: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_150, sub_50);  add_150 = sub_50 = None
    add_175: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Scalar(mul_291, 1);  mul_291 = None
    mul_292: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_35, add_175);  sigmoid_35 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_161: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_172: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_173: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_139)
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_36);  full_default_1 = None
    mul_312: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_139, sub_59);  add_139 = sub_59 = None
    add_176: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_176);  sigmoid_36 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_185: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_53: "f32[32, 256, 256]" = torch.ops.aten.permute.default(expand_15, [0, 2, 1]);  expand_15 = None
    permute_54: "f32[32, 128, 256]" = torch.ops.aten.permute.default(expand_16, [0, 2, 1]);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_9: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_58: "f32[31, 16]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_64: "f32[31, 16]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_66: "f32[32, 16, 256]" = torch.ops.aten.permute.default(expand_11, [0, 2, 1]);  expand_11 = None
    permute_67: "f32[32, 256, 16]" = torch.ops.aten.permute.default(expand_12, [0, 2, 1]);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    full_default_17: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_65: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_17, sigmoid_37);  full_default_17 = None
    mul_327: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sub_65);  add_132 = sub_65 = None
    add_179: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_327, 1);  mul_327 = None
    mul_328: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_37, add_179);  sigmoid_37 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_38: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    full_default_18: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_70: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_38)
    mul_339: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sub_70);  add_127 = sub_70 = None
    add_181: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_339, 1);  mul_339 = None
    mul_340: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_38, add_181);  sigmoid_38 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_209: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_121)
    full_default_19: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_75: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_19, sigmoid_39)
    mul_351: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_121, sub_75);  add_121 = sub_75 = None
    add_182: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_351, 1);  mul_351 = None
    mul_352: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_39, add_182);  sigmoid_39 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_74: "f32[32, 256, 256]" = torch.ops.aten.permute.default(expand_9, [0, 2, 1]);  expand_9 = None
    permute_75: "f32[32, 64, 256]" = torch.ops.aten.permute.default(expand_10, [0, 2, 1]);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_10: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_79: "f32[31, 16]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_85: "f32[31, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_87: "f32[32, 16, 256]" = torch.ops.aten.permute.default(expand_5, [0, 2, 1]);  expand_5 = None
    permute_88: "f32[32, 256, 16]" = torch.ops.aten.permute.default(expand_6, [0, 2, 1]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_114)
    sub_81: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_19, sigmoid_40);  full_default_19 = None
    mul_366: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_114, sub_81);  add_114 = sub_81 = None
    add_185: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_366, 1);  mul_366 = None
    mul_367: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_185);  sigmoid_40 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_41: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_109)
    sub_86: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_41);  full_default_18 = None
    mul_378: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_109, sub_86);  add_109 = sub_86 = None
    add_187: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_378, 1);  mul_378 = None
    mul_379: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_187);  sigmoid_41 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_245: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_257: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_270: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_271: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    full_default_29: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_101: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_29, sigmoid_43);  full_default_29 = None
    mul_415: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sub_101);  add_93 = sub_101 = None
    add_190: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_415, 1);  mul_415 = None
    mul_416: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_190);  sigmoid_43 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_282: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_283: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_44: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_88)
    full_default_30: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_106: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_30, sigmoid_44)
    mul_427: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_88, sub_106);  add_88 = sub_106 = None
    add_192: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_427, 1);  mul_427 = None
    mul_428: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_192);  sigmoid_44 = add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_294: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_295: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_31: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_46: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_77)
    sub_117: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_46);  full_default_31 = None
    mul_455: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_77, sub_117);  add_77 = sub_117 = None
    add_195: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_455, 1);  mul_455 = None
    mul_456: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_195);  sigmoid_46 = add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_47: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_72)
    sub_122: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_30, sigmoid_47);  full_default_30 = None
    mul_467: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, sub_122);  add_72 = sub_122 = None
    add_197: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_467, 1);  mul_467 = None
    mul_468: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_47, add_197);  sigmoid_47 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_333: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_345: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_358: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_359: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
    full_default_35: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_137: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_35, sigmoid_49);  full_default_35 = None
    mul_504: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sub_137);  add_56 = sub_137 = None
    add_200: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_504, 1);  mul_504 = None
    mul_505: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_200);  sigmoid_49 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_370: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_371: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_50: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_51)
    full_default_36: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_142: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_50)
    mul_516: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, sub_142);  add_51 = sub_142 = None
    add_202: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_516, 1);  mul_516 = None
    mul_517: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_202);  sigmoid_50 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_37: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_397: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_40)
    sub_153: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_52)
    mul_544: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, sub_153);  add_40 = sub_153 = None
    add_205: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_544, 1);  mul_544 = None
    mul_545: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_205);  sigmoid_52 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_409: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_53: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_35)
    sub_158: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_53);  full_default_36 = None
    mul_556: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_35, sub_158);  add_35 = sub_158 = None
    add_207: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_556, 1);  mul_556 = None
    mul_557: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_207);  sigmoid_53 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_421: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_432: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_433: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_446: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_447: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_55: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_19)
    sub_173: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_55);  full_default_37 = None
    mul_593: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, sub_173);  add_19 = sub_173 = None
    add_210: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_593, 1);  mul_593 = None
    mul_594: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_210);  sigmoid_55 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_458: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_459: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_14)
    full_default_42: "f32[8, 64, 128, 128]" = torch.ops.aten.full.default([8, 64, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_178: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_42, sigmoid_56);  full_default_42 = None
    mul_605: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, sub_178);  add_14 = sub_178 = None
    add_212: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_212);  sigmoid_56 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_470: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_471: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    full_default_43: "f32[8, 32, 128, 128]" = torch.ops.aten.full.default([8, 32, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_183: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_43, sigmoid_57);  full_default_43 = None
    mul_617: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sub_183);  add_9 = sub_183 = None
    add_213: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Scalar(mul_617, 1);  mul_617 = None
    mul_618: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_213);  sigmoid_57 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_482: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_483: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    full_default_44: "f32[8, 24, 128, 128]" = torch.ops.aten.full.default([8, 24, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_188: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_44, sigmoid_58);  full_default_44 = None
    mul_629: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sub_188);  add_4 = sub_188 = None
    add_214: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Scalar(mul_629, 1);  mul_629 = None
    mul_630: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_214);  sigmoid_58 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_494: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_495: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_107, add);  primals_107 = add = None
    copy__1: "f32[24]" = torch.ops.aten.copy_.default(primals_108, add_2);  primals_108 = add_2 = None
    copy__2: "f32[24]" = torch.ops.aten.copy_.default(primals_109, add_3);  primals_109 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_110, add_5);  primals_110 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_111, add_7);  primals_111 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_112, add_8);  primals_112 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_113, add_10);  primals_113 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_114, add_12);  primals_114 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_115, add_13);  primals_115 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_116, add_15);  primals_116 = add_15 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_117, add_17);  primals_117 = add_17 = None
    copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_118, add_18);  primals_118 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_119, add_20);  primals_119 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_120, add_22);  primals_120 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_121, add_23);  primals_121 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_122, add_25);  primals_122 = add_25 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_123, add_27);  primals_123 = add_27 = None
    copy__17: "f32[256]" = torch.ops.aten.copy_.default(primals_124, add_28);  primals_124 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_125, add_30);  primals_125 = add_30 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_126, add_32);  primals_126 = add_32 = None
    copy__20: "f32[256]" = torch.ops.aten.copy_.default(primals_127, add_33);  primals_127 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_128, add_36);  primals_128 = add_36 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_129, add_38);  primals_129 = add_38 = None
    copy__23: "f32[64]" = torch.ops.aten.copy_.default(primals_130, add_39);  primals_130 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_131, add_41);  primals_131 = add_41 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_132, add_43);  primals_132 = add_43 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_133, add_44);  primals_133 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_134, add_46);  primals_134 = add_46 = None
    copy__28: "f32[256]" = torch.ops.aten.copy_.default(primals_135, add_48);  primals_135 = add_48 = None
    copy__29: "f32[256]" = torch.ops.aten.copy_.default(primals_136, add_49);  primals_136 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_137, add_52);  primals_137 = add_52 = None
    copy__31: "f32[128]" = torch.ops.aten.copy_.default(primals_138, add_54);  primals_138 = add_54 = None
    copy__32: "f32[128]" = torch.ops.aten.copy_.default(primals_139, add_55);  primals_139 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_140, add_57);  primals_140 = add_57 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_141, add_59);  primals_141 = add_59 = None
    copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_142, add_60);  primals_142 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_143, add_62);  primals_143 = add_62 = None
    copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_144, add_64);  primals_144 = add_64 = None
    copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_145, add_65);  primals_145 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_146, add_67);  primals_146 = add_67 = None
    copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_147, add_69);  primals_147 = add_69 = None
    copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_148, add_70);  primals_148 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_149, add_73);  primals_149 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_150, add_75);  primals_150 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_151, add_76);  primals_151 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_152, add_78);  primals_152 = add_78 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_153, add_80);  primals_153 = add_80 = None
    copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_154, add_81);  primals_154 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_155, add_83);  primals_155 = add_83 = None
    copy__49: "f32[512]" = torch.ops.aten.copy_.default(primals_156, add_85);  primals_156 = add_85 = None
    copy__50: "f32[512]" = torch.ops.aten.copy_.default(primals_157, add_86);  primals_157 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_89);  primals_158 = add_89 = None
    copy__52: "f32[256]" = torch.ops.aten.copy_.default(primals_159, add_91);  primals_159 = add_91 = None
    copy__53: "f32[256]" = torch.ops.aten.copy_.default(primals_160, add_92);  primals_160 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_161, add_94);  primals_161 = add_94 = None
    copy__55: "f32[256]" = torch.ops.aten.copy_.default(primals_162, add_96);  primals_162 = add_96 = None
    copy__56: "f32[256]" = torch.ops.aten.copy_.default(primals_163, add_97);  primals_163 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_164, add_99);  primals_164 = add_99 = None
    copy__58: "f32[1024]" = torch.ops.aten.copy_.default(primals_165, add_101);  primals_165 = add_101 = None
    copy__59: "f32[1024]" = torch.ops.aten.copy_.default(primals_166, add_102);  primals_166 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_104);  primals_167 = add_104 = None
    copy__61: "f32[1024]" = torch.ops.aten.copy_.default(primals_168, add_106);  primals_168 = add_106 = None
    copy__62: "f32[1024]" = torch.ops.aten.copy_.default(primals_169, add_107);  primals_169 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_170, add_110);  primals_170 = add_110 = None
    copy__64: "f32[256]" = torch.ops.aten.copy_.default(primals_171, add_112);  primals_171 = add_112 = None
    copy__65: "f32[256]" = torch.ops.aten.copy_.default(primals_172, add_113);  primals_172 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_117);  primals_173 = add_117 = None
    copy__67: "f32[256]" = torch.ops.aten.copy_.default(primals_174, add_119);  primals_174 = add_119 = None
    copy__68: "f32[256]" = torch.ops.aten.copy_.default(primals_175, add_120);  primals_175 = add_120 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_176, add_122);  primals_176 = add_122 = None
    copy__70: "f32[1024]" = torch.ops.aten.copy_.default(primals_177, add_124);  primals_177 = add_124 = None
    copy__71: "f32[1024]" = torch.ops.aten.copy_.default(primals_178, add_125);  primals_178 = add_125 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_128);  primals_179 = add_128 = None
    copy__73: "f32[512]" = torch.ops.aten.copy_.default(primals_180, add_130);  primals_180 = add_130 = None
    copy__74: "f32[512]" = torch.ops.aten.copy_.default(primals_181, add_131);  primals_181 = add_131 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_182, add_135);  primals_182 = add_135 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_183, add_137);  primals_183 = add_137 = None
    copy__77: "f32[512]" = torch.ops.aten.copy_.default(primals_184, add_138);  primals_184 = add_138 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_140);  primals_185 = add_140 = None
    copy__79: "f32[2048]" = torch.ops.aten.copy_.default(primals_186, add_142);  primals_186 = add_142 = None
    copy__80: "f32[2048]" = torch.ops.aten.copy_.default(primals_187, add_143);  primals_187 = add_143 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_188, add_145);  primals_188 = add_145 = None
    copy__82: "f32[2048]" = torch.ops.aten.copy_.default(primals_189, add_147);  primals_189 = add_147 = None
    copy__83: "f32[2048]" = torch.ops.aten.copy_.default(primals_190, add_148);  primals_190 = add_148 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_151);  primals_191 = add_151 = None
    copy__85: "f32[512]" = torch.ops.aten.copy_.default(primals_192, add_153);  primals_192 = add_153 = None
    copy__86: "f32[512]" = torch.ops.aten.copy_.default(primals_193, add_154);  primals_193 = add_154 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_194, add_158);  primals_194 = add_158 = None
    copy__88: "f32[512]" = torch.ops.aten.copy_.default(primals_195, add_160);  primals_195 = add_160 = None
    copy__89: "f32[512]" = torch.ops.aten.copy_.default(primals_196, add_161);  primals_196 = add_161 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_163);  primals_197 = add_163 = None
    copy__91: "f32[2048]" = torch.ops.aten.copy_.default(primals_198, add_165);  primals_198 = add_165 = None
    copy__92: "f32[2048]" = torch.ops.aten.copy_.default(primals_199, add_166);  primals_199 = add_166 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_200, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, bmm_1, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, view_41, view_47, view_57, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, view_65, view_71, bmm_5, squeeze_88, mul_243, convolution_35, squeeze_91, view_82, permute_25, mul_253, unsqueeze_126, mul_265, unsqueeze_138, permute_32, permute_33, alias_8, permute_37, permute_43, permute_45, permute_46, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_53, permute_54, alias_9, permute_58, permute_64, permute_66, permute_67, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, unsqueeze_222, permute_74, permute_75, alias_10, permute_79, permute_85, permute_87, permute_88, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496]
    