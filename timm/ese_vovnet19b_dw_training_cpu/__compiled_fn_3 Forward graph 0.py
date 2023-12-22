from __future__ import annotations



def forward(self, primals_1: "f32[64]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[160]", primals_18: "f32[160]", primals_19: "f32[160]", primals_20: "f32[160]", primals_21: "f32[160]", primals_22: "f32[160]", primals_23: "f32[160]", primals_24: "f32[160]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[192]", primals_28: "f32[192]", primals_29: "f32[192]", primals_30: "f32[192]", primals_31: "f32[192]", primals_32: "f32[192]", primals_33: "f32[192]", primals_34: "f32[192]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[224]", primals_38: "f32[224]", primals_39: "f32[224]", primals_40: "f32[224]", primals_41: "f32[224]", primals_42: "f32[224]", primals_43: "f32[224]", primals_44: "f32[224]", primals_45: "f32[1024]", primals_46: "f32[1024]", primals_47: "f32[64, 3, 3, 3]", primals_48: "f32[64, 1, 3, 3]", primals_49: "f32[64, 64, 1, 1]", primals_50: "f32[64, 1, 3, 3]", primals_51: "f32[64, 64, 1, 1]", primals_52: "f32[128, 64, 1, 1]", primals_53: "f32[128, 1, 3, 3]", primals_54: "f32[128, 128, 1, 1]", primals_55: "f32[128, 1, 3, 3]", primals_56: "f32[128, 128, 1, 1]", primals_57: "f32[128, 1, 3, 3]", primals_58: "f32[128, 128, 1, 1]", primals_59: "f32[256, 448, 1, 1]", primals_60: "f32[256, 256, 1, 1]", primals_61: "f32[256]", primals_62: "f32[160, 256, 1, 1]", primals_63: "f32[160, 1, 3, 3]", primals_64: "f32[160, 160, 1, 1]", primals_65: "f32[160, 1, 3, 3]", primals_66: "f32[160, 160, 1, 1]", primals_67: "f32[160, 1, 3, 3]", primals_68: "f32[160, 160, 1, 1]", primals_69: "f32[512, 736, 1, 1]", primals_70: "f32[512, 512, 1, 1]", primals_71: "f32[512]", primals_72: "f32[192, 512, 1, 1]", primals_73: "f32[192, 1, 3, 3]", primals_74: "f32[192, 192, 1, 1]", primals_75: "f32[192, 1, 3, 3]", primals_76: "f32[192, 192, 1, 1]", primals_77: "f32[192, 1, 3, 3]", primals_78: "f32[192, 192, 1, 1]", primals_79: "f32[768, 1088, 1, 1]", primals_80: "f32[768, 768, 1, 1]", primals_81: "f32[768]", primals_82: "f32[224, 768, 1, 1]", primals_83: "f32[224, 1, 3, 3]", primals_84: "f32[224, 224, 1, 1]", primals_85: "f32[224, 1, 3, 3]", primals_86: "f32[224, 224, 1, 1]", primals_87: "f32[224, 1, 3, 3]", primals_88: "f32[224, 224, 1, 1]", primals_89: "f32[1024, 1440, 1, 1]", primals_90: "f32[1024, 1024, 1, 1]", primals_91: "f32[1024]", primals_92: "f32[1000, 1024]", primals_93: "f32[1000]", primals_94: "i64[]", primals_95: "f32[64]", primals_96: "f32[64]", primals_97: "i64[]", primals_98: "f32[64]", primals_99: "f32[64]", primals_100: "i64[]", primals_101: "f32[64]", primals_102: "f32[64]", primals_103: "i64[]", primals_104: "f32[128]", primals_105: "f32[128]", primals_106: "i64[]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "i64[]", primals_110: "f32[128]", primals_111: "f32[128]", primals_112: "i64[]", primals_113: "f32[128]", primals_114: "f32[128]", primals_115: "i64[]", primals_116: "f32[256]", primals_117: "f32[256]", primals_118: "i64[]", primals_119: "f32[160]", primals_120: "f32[160]", primals_121: "i64[]", primals_122: "f32[160]", primals_123: "f32[160]", primals_124: "i64[]", primals_125: "f32[160]", primals_126: "f32[160]", primals_127: "i64[]", primals_128: "f32[160]", primals_129: "f32[160]", primals_130: "i64[]", primals_131: "f32[512]", primals_132: "f32[512]", primals_133: "i64[]", primals_134: "f32[192]", primals_135: "f32[192]", primals_136: "i64[]", primals_137: "f32[192]", primals_138: "f32[192]", primals_139: "i64[]", primals_140: "f32[192]", primals_141: "f32[192]", primals_142: "i64[]", primals_143: "f32[192]", primals_144: "f32[192]", primals_145: "i64[]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "i64[]", primals_149: "f32[224]", primals_150: "f32[224]", primals_151: "i64[]", primals_152: "f32[224]", primals_153: "f32[224]", primals_154: "i64[]", primals_155: "f32[224]", primals_156: "f32[224]", primals_157: "i64[]", primals_158: "f32[224]", primals_159: "f32[224]", primals_160: "i64[]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_163, primals_47, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_94, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 64, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_48, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(convolution_1, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_97, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_3)
    mul_7: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_98, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_50, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(convolution_3, primals_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_100, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_5)
    mul_14: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_103, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_21: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[128]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
    add_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_105, 0.9)
    add_18: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_53, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_7: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(convolution_6, primals_54, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_106, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_9)
    mul_28: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
    add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
    add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_8: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_9: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(convolution_8, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_11)
    mul_35: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[128]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
    add_27: "f32[128]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[128]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[128]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
    add_28: "f32[128]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_57, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_11: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(convolution_10, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_112, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_13)
    mul_42: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
    add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[128]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[128]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
    add_33: "f32[128]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat: "f32[8, 448, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_4, relu_5, relu_6], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_59, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_115, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_15)
    mul_49: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
    add_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(relu_7, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_13: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_60, primals_61, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_40: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_13, 3)
    clamp_min: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_40, 0);  add_40 = None
    clamp_max: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    div: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
    mul_56: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(relu_7, div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_16: "f32[8, 256, 28, 28]" = max_pool2d_with_indices[0]
    getitem_17: "i64[8, 256, 28, 28]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(getitem_16, primals_62, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_118, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 160, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 160, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_19)
    mul_57: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_58: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_59: "f32[160]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
    add_43: "f32[160]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    squeeze_26: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_60: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_61: "f32[160]" = torch.ops.aten.mul.Tensor(mul_60, 0.1);  mul_60 = None
    mul_62: "f32[160]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
    add_44: "f32[160]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
    unsqueeze_32: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_63: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_33);  mul_57 = unsqueeze_33 = None
    unsqueeze_34: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_35);  mul_63 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 160, 28, 28]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_63, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_16: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(convolution_15, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_121, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 160, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 160, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
    mul_64: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_65: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_66: "f32[160]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
    add_48: "f32[160]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_29: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_67: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
    mul_68: "f32[160]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[160]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
    add_49: "f32[160]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_36: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_70: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
    unsqueeze_38: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 160, 28, 28]" = torch.ops.aten.relu.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_65, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_18: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(convolution_17, primals_66, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_124, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 160, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 160, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_23)
    mul_71: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_72: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_73: "f32[160]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
    add_53: "f32[160]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_32: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_74: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_75: "f32[160]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[160]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
    add_54: "f32[160]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    unsqueeze_40: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_77: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
    unsqueeze_42: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 160, 28, 28]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 160, 28, 28]" = torch.ops.aten.convolution.default(convolution_19, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_127, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 160, 1, 1]" = var_mean_11[0]
    getitem_25: "f32[1, 160, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_11: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
    mul_78: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_34: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_79: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_80: "f32[160]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
    add_58: "f32[160]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
    squeeze_35: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_81: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_82: "f32[160]" = torch.ops.aten.mul.Tensor(mul_81, 0.1);  mul_81 = None
    mul_83: "f32[160]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
    add_59: "f32[160]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    unsqueeze_44: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_84: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_45);  mul_78 = unsqueeze_45 = None
    unsqueeze_46: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_47);  mul_84 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 160, 28, 28]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_1: "f32[8, 736, 28, 28]" = torch.ops.aten.cat.default([getitem_16, relu_9, relu_10, relu_11], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_1, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_130, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1, 1]" = var_mean_12[0]
    getitem_27: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
    mul_85: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_86: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_87: "f32[512]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
    add_63: "f32[512]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
    squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_88: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(mul_88, 0.1);  mul_88 = None
    mul_90: "f32[512]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_91: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_49);  mul_85 = unsqueeze_49 = None
    unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_51);  mul_91 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_12, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_22: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_70, primals_71, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_66: "f32[8, 512, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3)
    clamp_min_1: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_1: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[8, 512, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    mul_92: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(relu_12, div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_28: "f32[8, 512, 14, 14]" = max_pool2d_with_indices_1[0]
    getitem_29: "i64[8, 512, 14, 14]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(getitem_28, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_133, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 192, 1, 1]" = var_mean_13[0]
    getitem_31: "f32[1, 192, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_13: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_31)
    mul_93: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_40: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_94: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_95: "f32[192]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
    add_69: "f32[192]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_41: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_96: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_97: "f32[192]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[192]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
    add_70: "f32[192]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_52: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_99: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
    unsqueeze_54: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_24: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_13, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_25: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(convolution_24, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_136, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 192, 1, 1]" = var_mean_14[0]
    getitem_33: "f32[1, 192, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_14: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_33)
    mul_100: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_43: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_101: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_102: "f32[192]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
    add_74: "f32[192]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_44: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_103: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_104: "f32[192]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[192]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
    add_75: "f32[192]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_56: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_106: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_57);  mul_100 = unsqueeze_57 = None
    unsqueeze_58: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_59);  mul_106 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_14, primals_75, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(convolution_26, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_139, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 192, 1, 1]" = var_mean_15[0]
    getitem_35: "f32[1, 192, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_15: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_35)
    mul_107: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_46: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_108: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_109: "f32[192]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
    add_79: "f32[192]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    squeeze_47: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_110: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_111: "f32[192]" = torch.ops.aten.mul.Tensor(mul_110, 0.1);  mul_110 = None
    mul_112: "f32[192]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
    add_80: "f32[192]" = torch.ops.aten.add.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
    unsqueeze_60: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_113: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_61);  mul_107 = unsqueeze_61 = None
    unsqueeze_62: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_63);  mul_113 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_77, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_29: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(convolution_28, primals_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_142, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 192, 1, 1]" = var_mean_16[0]
    getitem_37: "f32[1, 192, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_16: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_37)
    mul_114: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_49: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_115: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_116: "f32[192]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
    add_84: "f32[192]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    squeeze_50: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_117: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_118: "f32[192]" = torch.ops.aten.mul.Tensor(mul_117, 0.1);  mul_117 = None
    mul_119: "f32[192]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
    add_85: "f32[192]" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    unsqueeze_64: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_120: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_114, unsqueeze_65);  mul_114 = unsqueeze_65 = None
    unsqueeze_66: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_67);  mul_120 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_2: "f32[8, 1088, 14, 14]" = torch.ops.aten.cat.default([getitem_28, relu_14, relu_15, relu_16], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_2, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_145, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 768, 1, 1]" = var_mean_17[0]
    getitem_39: "f32[1, 768, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_88: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_17: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_17: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_39)
    mul_121: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_52: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_122: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_123: "f32[768]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
    add_89: "f32[768]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_53: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_124: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_125: "f32[768]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[768]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
    add_90: "f32[768]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    unsqueeze_68: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_127: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_69);  mul_121 = unsqueeze_69 = None
    unsqueeze_70: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_91: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_71);  mul_127 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 768, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(relu_17, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_31: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_80, primals_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_92: "f32[8, 768, 1, 1]" = torch.ops.aten.add.Tensor(convolution_31, 3)
    clamp_min_2: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_min.default(add_92, 0);  add_92 = None
    clamp_max_2: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[8, 768, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    mul_128: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(relu_17, div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_40: "f32[8, 768, 7, 7]" = max_pool2d_with_indices_2[0]
    getitem_41: "i64[8, 768, 7, 7]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(getitem_40, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_148, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 224, 1, 1]" = var_mean_18[0]
    getitem_43: "f32[1, 224, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_18: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_43)
    mul_129: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_55: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_130: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_131: "f32[224]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
    add_95: "f32[224]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    squeeze_56: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_132: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0025575447570332);  squeeze_56 = None
    mul_133: "f32[224]" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
    mul_134: "f32[224]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_96: "f32[224]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
    unsqueeze_72: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_135: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_73);  mul_129 = unsqueeze_73 = None
    unsqueeze_74: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_75);  mul_135 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 224, 7, 7]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_18, primals_83, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_34: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(convolution_33, primals_84, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 224, 1, 1]" = var_mean_19[0]
    getitem_45: "f32[1, 224, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_99: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_19: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_19: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
    mul_136: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_58: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_137: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_138: "f32[224]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
    add_100: "f32[224]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    squeeze_59: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_139: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0025575447570332);  squeeze_59 = None
    mul_140: "f32[224]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
    mul_141: "f32[224]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_101: "f32[224]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
    unsqueeze_76: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_142: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_77);  mul_136 = unsqueeze_77 = None
    unsqueeze_78: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_102: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_79);  mul_142 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 224, 7, 7]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_19, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_36: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(convolution_35, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_154, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 224, 1, 1]" = var_mean_20[0]
    getitem_47: "f32[1, 224, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_104: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_20: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_20: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_47)
    mul_143: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_61: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_144: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_145: "f32[224]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
    add_105: "f32[224]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    squeeze_62: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_146: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0025575447570332);  squeeze_62 = None
    mul_147: "f32[224]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
    mul_148: "f32[224]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_106: "f32[224]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    unsqueeze_80: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_149: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_81);  mul_143 = unsqueeze_81 = None
    unsqueeze_82: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_107: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_83);  mul_149 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 224, 7, 7]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_20, primals_87, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_38: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(convolution_37, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_157, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 224, 1, 1]" = var_mean_21[0]
    getitem_49: "f32[1, 224, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_109: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_21: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_21: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_49)
    mul_150: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_64: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_151: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_152: "f32[224]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
    add_110: "f32[224]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    squeeze_65: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_153: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0025575447570332);  squeeze_65 = None
    mul_154: "f32[224]" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
    mul_155: "f32[224]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_111: "f32[224]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    unsqueeze_84: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_156: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_85);  mul_150 = unsqueeze_85 = None
    unsqueeze_86: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_112: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_87);  mul_156 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 224, 7, 7]" = torch.ops.aten.relu.default(add_112);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_3: "f32[8, 1440, 7, 7]" = torch.ops.aten.cat.default([getitem_40, relu_19, relu_20, relu_21], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_3, primals_89, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_160, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1, 1]" = var_mean_22[0]
    getitem_51: "f32[1, 1024, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_114: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_22: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_51)
    mul_157: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_67: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_158: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_159: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
    add_115: "f32[1024]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    squeeze_68: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_160: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0025575447570332);  squeeze_68 = None
    mul_161: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_160, 0.1);  mul_160 = None
    mul_162: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_116: "f32[1024]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    unsqueeze_88: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_163: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_89);  mul_157 = unsqueeze_89 = None
    unsqueeze_90: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_117: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_91);  mul_163 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_117);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_40: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_90, primals_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    add_118: "f32[8, 1024, 1, 1]" = torch.ops.aten.add.Tensor(convolution_40, 3)
    clamp_min_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_min.default(add_118, 0);  add_118 = None
    clamp_max_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    mul_164: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(relu_22, div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_4: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(mul_164, [-1, -2], True);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.view.default(mean_4, [8, 1024]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 1024]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_93, clone, permute);  primals_93 = None
    permute_1: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    gt: "b8[8, 1024, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_40, -3.0)
    lt: "b8[8, 1024, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_40, 3.0);  convolution_40 = None
    bitwise_and: "b8[8, 1024, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_92: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_93: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
    unsqueeze_94: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_27: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_28: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    le_1: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_28, 0);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_105: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
    unsqueeze_106: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 3);  unsqueeze_105 = None
    unsqueeze_116: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_117: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
    unsqueeze_118: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_129: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_140: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_141: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    gt_1: "b8[8, 768, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_31, -3.0)
    lt_1: "b8[8, 768, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_31, 3.0);  convolution_31 = None
    bitwise_and_1: "b8[8, 768, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_1);  gt_1 = lt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_153: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_42: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_43: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_6: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_165: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    unsqueeze_176: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_177: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_189: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_201: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    gt_2: "b8[8, 512, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_22, -3.0)
    lt_2: "b8[8, 512, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_22, 3.0);  convolution_22 = None
    bitwise_and_2: "b8[8, 512, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_2);  gt_2 = lt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_213: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_57: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_58: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_11: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_225: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    unsqueeze_236: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_237: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_249: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_261: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    gt_3: "b8[8, 256, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_13, -3.0)
    lt_3: "b8[8, 256, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_13, 3.0);  convolution_13 = None
    bitwise_and_3: "b8[8, 256, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_3);  gt_3 = lt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_273: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_73: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_16: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_285: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    unsqueeze_296: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_297: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_333: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    unsqueeze_344: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_345: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_94, add);  primals_94 = add = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_95, add_2);  primals_95 = add_2 = None
    copy__2: "f32[64]" = torch.ops.aten.copy_.default(primals_96, add_3);  primals_96 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_97, add_5);  primals_97 = add_5 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_98, add_7);  primals_98 = add_7 = None
    copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_99, add_8);  primals_99 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_100, add_10);  primals_100 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_101, add_12);  primals_101 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_102, add_13);  primals_102 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_103, add_15);  primals_103 = add_15 = None
    copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_104, add_17);  primals_104 = add_17 = None
    copy__11: "f32[128]" = torch.ops.aten.copy_.default(primals_105, add_18);  primals_105 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_106, add_20);  primals_106 = add_20 = None
    copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_107, add_22);  primals_107 = add_22 = None
    copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_108, add_23);  primals_108 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_25);  primals_109 = add_25 = None
    copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_110, add_27);  primals_110 = add_27 = None
    copy__17: "f32[128]" = torch.ops.aten.copy_.default(primals_111, add_28);  primals_111 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_112, add_30);  primals_112 = add_30 = None
    copy__19: "f32[128]" = torch.ops.aten.copy_.default(primals_113, add_32);  primals_113 = add_32 = None
    copy__20: "f32[128]" = torch.ops.aten.copy_.default(primals_114, add_33);  primals_114 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_115, add_35);  primals_115 = add_35 = None
    copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_116, add_37);  primals_116 = add_37 = None
    copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_117, add_38);  primals_117 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_118, add_41);  primals_118 = add_41 = None
    copy__25: "f32[160]" = torch.ops.aten.copy_.default(primals_119, add_43);  primals_119 = add_43 = None
    copy__26: "f32[160]" = torch.ops.aten.copy_.default(primals_120, add_44);  primals_120 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_121, add_46);  primals_121 = add_46 = None
    copy__28: "f32[160]" = torch.ops.aten.copy_.default(primals_122, add_48);  primals_122 = add_48 = None
    copy__29: "f32[160]" = torch.ops.aten.copy_.default(primals_123, add_49);  primals_123 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_124, add_51);  primals_124 = add_51 = None
    copy__31: "f32[160]" = torch.ops.aten.copy_.default(primals_125, add_53);  primals_125 = add_53 = None
    copy__32: "f32[160]" = torch.ops.aten.copy_.default(primals_126, add_54);  primals_126 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_127, add_56);  primals_127 = add_56 = None
    copy__34: "f32[160]" = torch.ops.aten.copy_.default(primals_128, add_58);  primals_128 = add_58 = None
    copy__35: "f32[160]" = torch.ops.aten.copy_.default(primals_129, add_59);  primals_129 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_130, add_61);  primals_130 = add_61 = None
    copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_131, add_63);  primals_131 = add_63 = None
    copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_132, add_64);  primals_132 = add_64 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_133, add_67);  primals_133 = add_67 = None
    copy__40: "f32[192]" = torch.ops.aten.copy_.default(primals_134, add_69);  primals_134 = add_69 = None
    copy__41: "f32[192]" = torch.ops.aten.copy_.default(primals_135, add_70);  primals_135 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_136, add_72);  primals_136 = add_72 = None
    copy__43: "f32[192]" = torch.ops.aten.copy_.default(primals_137, add_74);  primals_137 = add_74 = None
    copy__44: "f32[192]" = torch.ops.aten.copy_.default(primals_138, add_75);  primals_138 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_139, add_77);  primals_139 = add_77 = None
    copy__46: "f32[192]" = torch.ops.aten.copy_.default(primals_140, add_79);  primals_140 = add_79 = None
    copy__47: "f32[192]" = torch.ops.aten.copy_.default(primals_141, add_80);  primals_141 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_142, add_82);  primals_142 = add_82 = None
    copy__49: "f32[192]" = torch.ops.aten.copy_.default(primals_143, add_84);  primals_143 = add_84 = None
    copy__50: "f32[192]" = torch.ops.aten.copy_.default(primals_144, add_85);  primals_144 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_145, add_87);  primals_145 = add_87 = None
    copy__52: "f32[768]" = torch.ops.aten.copy_.default(primals_146, add_89);  primals_146 = add_89 = None
    copy__53: "f32[768]" = torch.ops.aten.copy_.default(primals_147, add_90);  primals_147 = add_90 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_148, add_93);  primals_148 = add_93 = None
    copy__55: "f32[224]" = torch.ops.aten.copy_.default(primals_149, add_95);  primals_149 = add_95 = None
    copy__56: "f32[224]" = torch.ops.aten.copy_.default(primals_150, add_96);  primals_150 = add_96 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_98);  primals_151 = add_98 = None
    copy__58: "f32[224]" = torch.ops.aten.copy_.default(primals_152, add_100);  primals_152 = add_100 = None
    copy__59: "f32[224]" = torch.ops.aten.copy_.default(primals_153, add_101);  primals_153 = add_101 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_154, add_103);  primals_154 = add_103 = None
    copy__61: "f32[224]" = torch.ops.aten.copy_.default(primals_155, add_105);  primals_155 = add_105 = None
    copy__62: "f32[224]" = torch.ops.aten.copy_.default(primals_156, add_106);  primals_156 = add_106 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_157, add_108);  primals_157 = add_108 = None
    copy__64: "f32[224]" = torch.ops.aten.copy_.default(primals_158, add_110);  primals_158 = add_110 = None
    copy__65: "f32[224]" = torch.ops.aten.copy_.default(primals_159, add_111);  primals_159 = add_111 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_160, add_113);  primals_160 = add_113 = None
    copy__67: "f32[1024]" = torch.ops.aten.copy_.default(primals_161, add_115);  primals_161 = add_115 = None
    copy__68: "f32[1024]" = torch.ops.aten.copy_.default(primals_162, add_116);  primals_162 = add_116 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_163, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean, div, mul_56, getitem_16, getitem_17, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_1, div_1, mul_92, getitem_28, getitem_29, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_2, div_2, mul_128, getitem_40, getitem_41, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_3, div_3, clone, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358]
    