from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[128]"; primals_8: "f32[128]"; primals_9: "f32[128]"; primals_10: "f32[128]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[128]"; primals_14: "f32[128]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[160]"; primals_18: "f32[160]"; primals_19: "f32[160]"; primals_20: "f32[160]"; primals_21: "f32[160]"; primals_22: "f32[160]"; primals_23: "f32[160]"; primals_24: "f32[160]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[192]"; primals_28: "f32[192]"; primals_29: "f32[192]"; primals_30: "f32[192]"; primals_31: "f32[192]"; primals_32: "f32[192]"; primals_33: "f32[192]"; primals_34: "f32[192]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[224]"; primals_38: "f32[224]"; primals_39: "f32[224]"; primals_40: "f32[224]"; primals_41: "f32[224]"; primals_42: "f32[224]"; primals_43: "f32[224]"; primals_44: "f32[224]"; primals_45: "f32[1024]"; primals_46: "f32[1024]"; primals_47: "f32[64, 3, 3, 3]"; primals_48: "f32[64, 1, 3, 3]"; primals_49: "f32[64, 64, 1, 1]"; primals_50: "f32[64, 1, 3, 3]"; primals_51: "f32[64, 64, 1, 1]"; primals_52: "f32[128, 64, 1, 1]"; primals_53: "f32[128, 1, 3, 3]"; primals_54: "f32[128, 128, 1, 1]"; primals_55: "f32[128, 1, 3, 3]"; primals_56: "f32[128, 128, 1, 1]"; primals_57: "f32[128, 1, 3, 3]"; primals_58: "f32[128, 128, 1, 1]"; primals_59: "f32[256, 448, 1, 1]"; primals_60: "f32[256, 256, 1, 1]"; primals_61: "f32[256]"; primals_62: "f32[160, 256, 1, 1]"; primals_63: "f32[160, 1, 3, 3]"; primals_64: "f32[160, 160, 1, 1]"; primals_65: "f32[160, 1, 3, 3]"; primals_66: "f32[160, 160, 1, 1]"; primals_67: "f32[160, 1, 3, 3]"; primals_68: "f32[160, 160, 1, 1]"; primals_69: "f32[512, 736, 1, 1]"; primals_70: "f32[512, 512, 1, 1]"; primals_71: "f32[512]"; primals_72: "f32[192, 512, 1, 1]"; primals_73: "f32[192, 1, 3, 3]"; primals_74: "f32[192, 192, 1, 1]"; primals_75: "f32[192, 1, 3, 3]"; primals_76: "f32[192, 192, 1, 1]"; primals_77: "f32[192, 1, 3, 3]"; primals_78: "f32[192, 192, 1, 1]"; primals_79: "f32[768, 1088, 1, 1]"; primals_80: "f32[768, 768, 1, 1]"; primals_81: "f32[768]"; primals_82: "f32[224, 768, 1, 1]"; primals_83: "f32[224, 1, 3, 3]"; primals_84: "f32[224, 224, 1, 1]"; primals_85: "f32[224, 1, 3, 3]"; primals_86: "f32[224, 224, 1, 1]"; primals_87: "f32[224, 1, 3, 3]"; primals_88: "f32[224, 224, 1, 1]"; primals_89: "f32[1024, 1440, 1, 1]"; primals_90: "f32[1024, 1024, 1, 1]"; primals_91: "f32[1024]"; primals_92: "f32[1000, 1024]"; primals_93: "f32[1000]"; primals_94: "i64[]"; primals_95: "f32[64]"; primals_96: "f32[64]"; primals_97: "i64[]"; primals_98: "f32[64]"; primals_99: "f32[64]"; primals_100: "i64[]"; primals_101: "f32[64]"; primals_102: "f32[64]"; primals_103: "i64[]"; primals_104: "f32[128]"; primals_105: "f32[128]"; primals_106: "i64[]"; primals_107: "f32[128]"; primals_108: "f32[128]"; primals_109: "i64[]"; primals_110: "f32[128]"; primals_111: "f32[128]"; primals_112: "i64[]"; primals_113: "f32[128]"; primals_114: "f32[128]"; primals_115: "i64[]"; primals_116: "f32[256]"; primals_117: "f32[256]"; primals_118: "i64[]"; primals_119: "f32[160]"; primals_120: "f32[160]"; primals_121: "i64[]"; primals_122: "f32[160]"; primals_123: "f32[160]"; primals_124: "i64[]"; primals_125: "f32[160]"; primals_126: "f32[160]"; primals_127: "i64[]"; primals_128: "f32[160]"; primals_129: "f32[160]"; primals_130: "i64[]"; primals_131: "f32[512]"; primals_132: "f32[512]"; primals_133: "i64[]"; primals_134: "f32[192]"; primals_135: "f32[192]"; primals_136: "i64[]"; primals_137: "f32[192]"; primals_138: "f32[192]"; primals_139: "i64[]"; primals_140: "f32[192]"; primals_141: "f32[192]"; primals_142: "i64[]"; primals_143: "f32[192]"; primals_144: "f32[192]"; primals_145: "i64[]"; primals_146: "f32[768]"; primals_147: "f32[768]"; primals_148: "i64[]"; primals_149: "f32[224]"; primals_150: "f32[224]"; primals_151: "i64[]"; primals_152: "f32[224]"; primals_153: "f32[224]"; primals_154: "i64[]"; primals_155: "f32[224]"; primals_156: "f32[224]"; primals_157: "i64[]"; primals_158: "f32[224]"; primals_159: "f32[224]"; primals_160: "i64[]"; primals_161: "f32[1024]"; primals_162: "f32[1024]"; primals_163: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
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
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1024, 7, 7]);  view_2 = None
    div_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_165: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_4, relu_22)
    mul_166: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_4, div_3);  div_4 = div_3 = None
    sum_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2, 3], True);  mul_165 = None
    gt: "b8[8, 1024, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_40, -3.0)
    lt: "b8[8, 1024, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_40, 3.0);  convolution_40 = None
    bitwise_and: "b8[8, 1024, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
    mul_167: "f32[8, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(sum_2, 0.16666666666666666);  sum_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 1024, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_167, scalar_tensor);  bitwise_and = mul_167 = scalar_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_backward = torch.ops.aten.convolution_backward.default(where, mean_3, primals_90, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where = mean_3 = primals_90 = None
    getitem_52: "f32[8, 1024, 1, 1]" = convolution_backward[0]
    getitem_53: "f32[1024, 1024, 1, 1]" = convolution_backward[1]
    getitem_54: "f32[1024]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(getitem_52, [8, 1024, 7, 7]);  getitem_52 = None
    div_5: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_119: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_166, div_5);  mul_166 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_24: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_25: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    le: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_25, 0);  alias_25 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor_1, add_119);  le = scalar_tensor_1 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_92: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_93: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
    unsqueeze_94: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_23: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_94)
    mul_168: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_23);  sub_23 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 2, 3]);  mul_168 = None
    mul_169: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_95: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_96: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
    unsqueeze_97: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
    mul_170: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_171: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_172: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    unsqueeze_98: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
    unsqueeze_99: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
    unsqueeze_100: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
    mul_173: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_101: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
    unsqueeze_102: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
    unsqueeze_103: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
    sub_24: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_94);  convolution_39 = unsqueeze_94 = None
    mul_174: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_100);  sub_24 = unsqueeze_100 = None
    sub_25: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_174);  where_1 = mul_174 = None
    sub_26: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_25, unsqueeze_97);  sub_25 = unsqueeze_97 = None
    mul_175: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_103);  sub_26 = unsqueeze_103 = None
    mul_176: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_67);  sum_4 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_175, cat_3, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = cat_3 = primals_89 = None
    getitem_55: "f32[8, 1440, 7, 7]" = convolution_backward_1[0]
    getitem_56: "f32[1024, 1440, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_1: "f32[8, 768, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 0, 768)
    slice_2: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 768, 992)
    slice_3: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 992, 1216)
    slice_4: "f32[8, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_55, 1, 1216, 1440);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_27: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_28: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    le_1: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_28, 0);  alias_28 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_2, slice_4);  le_1 = scalar_tensor_2 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_105: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
    unsqueeze_106: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 3);  unsqueeze_105 = None
    sum_5: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_27: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_106)
    mul_177: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_27);  sub_27 = None
    sum_6: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3]);  mul_177 = None
    mul_178: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_107: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_178, 0);  mul_178 = None
    unsqueeze_108: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
    unsqueeze_109: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
    mul_179: "f32[224]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_180: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_181: "f32[224]" = torch.ops.aten.mul.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_110: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
    unsqueeze_111: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
    unsqueeze_112: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
    mul_182: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_113: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
    unsqueeze_114: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    unsqueeze_115: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
    sub_28: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_106);  convolution_38 = unsqueeze_106 = None
    mul_183: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_112);  sub_28 = unsqueeze_112 = None
    sub_29: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_183);  where_2 = mul_183 = None
    sub_30: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_29, unsqueeze_109);  sub_29 = unsqueeze_109 = None
    mul_184: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_115);  sub_30 = unsqueeze_115 = None
    mul_185: "f32[224]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_64);  sum_6 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_184, convolution_37, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_184 = convolution_37 = primals_88 = None
    getitem_58: "f32[8, 224, 7, 7]" = convolution_backward_2[0]
    getitem_59: "f32[224, 224, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_58, relu_20, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_58 = primals_87 = None
    getitem_61: "f32[8, 224, 7, 7]" = convolution_backward_3[0]
    getitem_62: "f32[224, 1, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_120: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_61);  slice_3 = getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_30: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_31: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_2: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_3, add_120);  le_2 = scalar_tensor_3 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_116: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_117: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
    unsqueeze_118: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
    sum_7: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_31: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_118)
    mul_186: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_31);  sub_31 = None
    sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 2, 3]);  mul_186 = None
    mul_187: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_119: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_187, 0);  mul_187 = None
    unsqueeze_120: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    unsqueeze_121: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
    mul_188: "f32[224]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_189: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_190: "f32[224]" = torch.ops.aten.mul.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_122: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
    unsqueeze_123: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
    unsqueeze_124: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
    mul_191: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_125: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_191, 0);  mul_191 = None
    unsqueeze_126: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    unsqueeze_127: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
    sub_32: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_118);  convolution_36 = unsqueeze_118 = None
    mul_192: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_124);  sub_32 = unsqueeze_124 = None
    sub_33: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_192);  where_3 = mul_192 = None
    sub_34: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_33, unsqueeze_121);  sub_33 = unsqueeze_121 = None
    mul_193: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_127);  sub_34 = unsqueeze_127 = None
    mul_194: "f32[224]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_61);  sum_8 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_193, convolution_35, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_193 = convolution_35 = primals_86 = None
    getitem_64: "f32[8, 224, 7, 7]" = convolution_backward_4[0]
    getitem_65: "f32[224, 224, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_64, relu_19, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_64 = primals_85 = None
    getitem_67: "f32[8, 224, 7, 7]" = convolution_backward_5[0]
    getitem_68: "f32[224, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_121: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_67);  slice_2 = getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_33: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_34: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    le_3: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_34, 0);  alias_34 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_4, add_121);  le_3 = scalar_tensor_4 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_129: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    sum_9: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_35: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130)
    mul_195: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_35);  sub_35 = None
    sum_10: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3]);  mul_195 = None
    mul_196: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_131: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
    unsqueeze_132: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    mul_197: "f32[224]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_198: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_199: "f32[224]" = torch.ops.aten.mul.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    unsqueeze_134: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
    unsqueeze_135: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    mul_200: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_137: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_200, 0);  mul_200 = None
    unsqueeze_138: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    sub_36: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
    mul_201: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_136);  sub_36 = unsqueeze_136 = None
    sub_37: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_201);  where_4 = mul_201 = None
    sub_38: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_133);  sub_37 = unsqueeze_133 = None
    mul_202: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_139);  sub_38 = unsqueeze_139 = None
    mul_203: "f32[224]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_58);  sum_10 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_202, convolution_33, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_202 = convolution_33 = primals_84 = None
    getitem_70: "f32[8, 224, 7, 7]" = convolution_backward_6[0]
    getitem_71: "f32[224, 224, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_70, relu_18, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False]);  getitem_70 = primals_83 = None
    getitem_73: "f32[8, 224, 7, 7]" = convolution_backward_7[0]
    getitem_74: "f32[224, 1, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_36: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_37: "f32[8, 224, 7, 7]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_4: "b8[8, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 224, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_5, getitem_73);  le_4 = scalar_tensor_5 = getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_140: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_141: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    sum_11: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_39: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_142)
    mul_204: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_39);  sub_39 = None
    sum_12: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_205: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_143: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_205, 0);  mul_205 = None
    unsqueeze_144: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    mul_206: "f32[224]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_207: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_208: "f32[224]" = torch.ops.aten.mul.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    unsqueeze_146: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
    unsqueeze_147: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
    unsqueeze_148: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
    mul_209: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_149: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_150: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    unsqueeze_151: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
    sub_40: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_142);  convolution_32 = unsqueeze_142 = None
    mul_210: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_148);  sub_40 = unsqueeze_148 = None
    sub_41: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_210);  where_5 = mul_210 = None
    sub_42: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_145);  sub_41 = unsqueeze_145 = None
    mul_211: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_151);  sub_42 = unsqueeze_151 = None
    mul_212: "f32[224]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_55);  sum_12 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_211, getitem_40, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = getitem_40 = primals_82 = None
    getitem_76: "f32[8, 768, 7, 7]" = convolution_backward_8[0]
    getitem_77: "f32[224, 768, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_122: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_76);  slice_1 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward: "f32[8, 768, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_122, mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_41);  add_122 = mul_128 = getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_213: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, relu_17)
    mul_214: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, div_2);  max_pool2d_with_indices_backward = div_2 = None
    sum_13: "f32[8, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2, 3], True);  mul_213 = None
    gt_1: "b8[8, 768, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_31, -3.0)
    lt_1: "b8[8, 768, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_31, 3.0);  convolution_31 = None
    bitwise_and_1: "b8[8, 768, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_1);  gt_1 = lt_1 = None
    mul_215: "f32[8, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, 0.16666666666666666);  sum_13 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 768, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_215, scalar_tensor_6);  bitwise_and_1 = mul_215 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_6, mean_2, primals_80, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_6 = mean_2 = primals_80 = None
    getitem_79: "f32[8, 768, 1, 1]" = convolution_backward_9[0]
    getitem_80: "f32[768, 768, 1, 1]" = convolution_backward_9[1]
    getitem_81: "f32[768]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 768, 14, 14]" = torch.ops.aten.expand.default(getitem_79, [8, 768, 14, 14]);  getitem_79 = None
    div_6: "f32[8, 768, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_123: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_214, div_6);  mul_214 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_39: "f32[8, 768, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_40: "f32[8, 768, 14, 14]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_5: "b8[8, 768, 14, 14]" = torch.ops.aten.le.Scalar(alias_40, 0);  alias_40 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[8, 768, 14, 14]" = torch.ops.aten.where.self(le_5, scalar_tensor_7, add_123);  le_5 = scalar_tensor_7 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_153: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_43: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_154)
    mul_216: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_43);  sub_43 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3]);  mul_216 = None
    mul_217: "f32[768]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    unsqueeze_155: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_156: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_218: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
    mul_219: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    unsqueeze_158: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_159: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_221: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_161: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
    unsqueeze_162: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    sub_44: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_154);  convolution_30 = unsqueeze_154 = None
    mul_222: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_160);  sub_44 = unsqueeze_160 = None
    sub_45: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_222);  where_7 = mul_222 = None
    sub_46: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_157);  sub_45 = unsqueeze_157 = None
    mul_223: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_163);  sub_46 = unsqueeze_163 = None
    mul_224: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_52);  sum_15 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_223, cat_2, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_223 = cat_2 = primals_79 = None
    getitem_82: "f32[8, 1088, 14, 14]" = convolution_backward_10[0]
    getitem_83: "f32[768, 1088, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_5: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 0, 512)
    slice_6: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 512, 704)
    slice_7: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 704, 896)
    slice_8: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_82, 1, 896, 1088);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_42: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_43: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_6: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_6, scalar_tensor_8, slice_8);  le_6 = scalar_tensor_8 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_165: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_47: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_166)
    mul_225: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_47);  sub_47 = None
    sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
    mul_226: "f32[192]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
    unsqueeze_167: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
    unsqueeze_168: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_227: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_228: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_229: "f32[192]" = torch.ops.aten.mul.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    unsqueeze_170: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_229, 0);  mul_229 = None
    unsqueeze_171: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_230: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_173: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_174: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    sub_48: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_166);  convolution_29 = unsqueeze_166 = None
    mul_231: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_172);  sub_48 = unsqueeze_172 = None
    sub_49: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_231);  where_8 = mul_231 = None
    sub_50: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_169);  sub_49 = unsqueeze_169 = None
    mul_232: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_175);  sub_50 = unsqueeze_175 = None
    mul_233: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_49);  sum_17 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_232, convolution_28, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_232 = convolution_28 = primals_78 = None
    getitem_85: "f32[8, 192, 14, 14]" = convolution_backward_11[0]
    getitem_86: "f32[192, 192, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(getitem_85, relu_15, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_85 = primals_77 = None
    getitem_88: "f32[8, 192, 14, 14]" = convolution_backward_12[0]
    getitem_89: "f32[192, 1, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_124: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_7, getitem_88);  slice_7 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_45: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_46: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_7: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_7, scalar_tensor_9, add_124);  le_7 = scalar_tensor_9 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_177: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    sum_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_51: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_178)
    mul_234: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_51);  sub_51 = None
    sum_19: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 2, 3]);  mul_234 = None
    mul_235: "f32[192]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_179: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
    unsqueeze_180: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_236: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_237: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_238: "f32[192]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_182: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_183: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_239: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_185: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
    unsqueeze_186: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    sub_52: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_178);  convolution_27 = unsqueeze_178 = None
    mul_240: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_184);  sub_52 = unsqueeze_184 = None
    sub_53: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_240);  where_9 = mul_240 = None
    sub_54: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_181);  sub_53 = unsqueeze_181 = None
    mul_241: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_187);  sub_54 = unsqueeze_187 = None
    mul_242: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_46);  sum_19 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_241, convolution_26, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = convolution_26 = primals_76 = None
    getitem_91: "f32[8, 192, 14, 14]" = convolution_backward_13[0]
    getitem_92: "f32[192, 192, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(getitem_91, relu_14, primals_75, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_91 = primals_75 = None
    getitem_94: "f32[8, 192, 14, 14]" = convolution_backward_14[0]
    getitem_95: "f32[192, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_125: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_6, getitem_94);  slice_6 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_49: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_8: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_10, add_125);  le_8 = scalar_tensor_10 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_189: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    sum_20: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_55: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_190)
    mul_243: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_55);  sub_55 = None
    sum_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 2, 3]);  mul_243 = None
    mul_244: "f32[192]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_191: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
    unsqueeze_192: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_245: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_246: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_247: "f32[192]" = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    unsqueeze_194: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_247, 0);  mul_247 = None
    unsqueeze_195: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_248: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_197: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_248, 0);  mul_248 = None
    unsqueeze_198: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    sub_56: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_190);  convolution_25 = unsqueeze_190 = None
    mul_249: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_196);  sub_56 = unsqueeze_196 = None
    sub_57: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_249);  where_10 = mul_249 = None
    sub_58: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_193);  sub_57 = unsqueeze_193 = None
    mul_250: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_199);  sub_58 = unsqueeze_199 = None
    mul_251: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_43);  sum_21 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_250, convolution_24, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_250 = convolution_24 = primals_74 = None
    getitem_97: "f32[8, 192, 14, 14]" = convolution_backward_15[0]
    getitem_98: "f32[192, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(getitem_97, relu_13, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  getitem_97 = primals_73 = None
    getitem_100: "f32[8, 192, 14, 14]" = convolution_backward_16[0]
    getitem_101: "f32[192, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_51: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_52: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_9: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_9, scalar_tensor_11, getitem_100);  le_9 = scalar_tensor_11 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_201: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_59: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_202)
    mul_252: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_59);  sub_59 = None
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_253: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_203: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
    unsqueeze_204: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_254: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_255: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_256: "f32[192]" = torch.ops.aten.mul.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    unsqueeze_206: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
    unsqueeze_207: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_257: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_209: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_210: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sub_60: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_202);  convolution_23 = unsqueeze_202 = None
    mul_258: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_208);  sub_60 = unsqueeze_208 = None
    sub_61: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_258);  where_11 = mul_258 = None
    sub_62: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_205);  sub_61 = unsqueeze_205 = None
    mul_259: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_211);  sub_62 = unsqueeze_211 = None
    mul_260: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_40);  sum_23 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_259, getitem_28, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_259 = getitem_28 = primals_72 = None
    getitem_103: "f32[8, 512, 14, 14]" = convolution_backward_17[0]
    getitem_104: "f32[192, 512, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_126: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, getitem_103);  slice_5 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_1: "f32[8, 512, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_126, mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_29);  add_126 = mul_92 = getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_261: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, relu_12)
    mul_262: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_1, div_1);  max_pool2d_with_indices_backward_1 = div_1 = None
    sum_24: "f32[8, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2, 3], True);  mul_261 = None
    gt_2: "b8[8, 512, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_22, -3.0)
    lt_2: "b8[8, 512, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_22, 3.0);  convolution_22 = None
    bitwise_and_2: "b8[8, 512, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_2);  gt_2 = lt_2 = None
    mul_263: "f32[8, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, 0.16666666666666666);  sum_24 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[8, 512, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_263, scalar_tensor_12);  bitwise_and_2 = mul_263 = scalar_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_12, mean_1, primals_70, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_12 = mean_1 = primals_70 = None
    getitem_106: "f32[8, 512, 1, 1]" = convolution_backward_18[0]
    getitem_107: "f32[512, 512, 1, 1]" = convolution_backward_18[1]
    getitem_108: "f32[512]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_106, [8, 512, 28, 28]);  getitem_106 = None
    div_7: "f32[8, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_3, 784);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_127: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_262, div_7);  mul_262 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_55: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_10: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_10, scalar_tensor_13, add_127);  le_10 = scalar_tensor_13 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_213: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_63: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_214)
    mul_264: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_13, sub_63);  sub_63 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
    mul_265: "f32[512]" = torch.ops.aten.mul.Tensor(sum_25, 0.00015943877551020407)
    unsqueeze_215: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_216: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_266: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00015943877551020407)
    mul_267: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_218: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_219: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_221: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_222: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    sub_64: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_214);  convolution_21 = unsqueeze_214 = None
    mul_270: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_220);  sub_64 = unsqueeze_220 = None
    sub_65: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_13, mul_270);  where_13 = mul_270 = None
    sub_66: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_217);  sub_65 = unsqueeze_217 = None
    mul_271: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_223);  sub_66 = unsqueeze_223 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_37);  sum_26 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_271, cat_1, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_271 = cat_1 = primals_69 = None
    getitem_109: "f32[8, 736, 28, 28]" = convolution_backward_19[0]
    getitem_110: "f32[512, 736, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_9: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 0, 256)
    slice_10: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 256, 416)
    slice_11: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 416, 576)
    slice_12: "f32[8, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_109, 1, 576, 736);  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_57: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_58: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_11: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_11, scalar_tensor_14, slice_12);  le_11 = scalar_tensor_14 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_225: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    sum_27: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_67: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_226)
    mul_273: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_14, sub_67);  sub_67 = None
    sum_28: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3]);  mul_273 = None
    mul_274: "f32[160]" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
    unsqueeze_227: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_228: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_275: "f32[160]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
    mul_276: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_277: "f32[160]" = torch.ops.aten.mul.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    unsqueeze_230: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
    unsqueeze_231: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_278: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_233: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
    unsqueeze_234: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    sub_68: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_226);  convolution_20 = unsqueeze_226 = None
    mul_279: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_232);  sub_68 = unsqueeze_232 = None
    sub_69: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_14, mul_279);  where_14 = mul_279 = None
    sub_70: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_229);  sub_69 = unsqueeze_229 = None
    mul_280: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_235);  sub_70 = unsqueeze_235 = None
    mul_281: "f32[160]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_34);  sum_28 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_280, convolution_19, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = convolution_19 = primals_68 = None
    getitem_112: "f32[8, 160, 28, 28]" = convolution_backward_20[0]
    getitem_113: "f32[160, 160, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_112, relu_10, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_112 = primals_67 = None
    getitem_115: "f32[8, 160, 28, 28]" = convolution_backward_21[0]
    getitem_116: "f32[160, 1, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_128: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_11, getitem_115);  slice_11 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_60: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_61: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_12: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_12, scalar_tensor_15, add_128);  le_12 = scalar_tensor_15 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_237: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    sum_29: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_71: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_238)
    mul_282: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_15, sub_71);  sub_71 = None
    sum_30: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
    mul_283: "f32[160]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
    unsqueeze_239: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_240: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_284: "f32[160]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_242: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_243: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_287: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_245: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_246: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    sub_72: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_238);  convolution_18 = unsqueeze_238 = None
    mul_288: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_244);  sub_72 = unsqueeze_244 = None
    sub_73: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_15, mul_288);  where_15 = mul_288 = None
    sub_74: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_241);  sub_73 = unsqueeze_241 = None
    mul_289: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_247);  sub_74 = unsqueeze_247 = None
    mul_290: "f32[160]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_31);  sum_30 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_289, convolution_17, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = convolution_17 = primals_66 = None
    getitem_118: "f32[8, 160, 28, 28]" = convolution_backward_22[0]
    getitem_119: "f32[160, 160, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_118, relu_9, primals_65, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_118 = primals_65 = None
    getitem_121: "f32[8, 160, 28, 28]" = convolution_backward_23[0]
    getitem_122: "f32[160, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_129: "f32[8, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_10, getitem_121);  slice_10 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_63: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_64: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_13: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_13, scalar_tensor_16, add_129);  le_13 = scalar_tensor_16 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_249: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    sum_31: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_75: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_250)
    mul_291: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_75);  sub_75 = None
    sum_32: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
    mul_292: "f32[160]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
    unsqueeze_251: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_252: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_293: "f32[160]" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
    mul_294: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_295: "f32[160]" = torch.ops.aten.mul.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    unsqueeze_254: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_255: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_296: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_257: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
    unsqueeze_258: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sub_76: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_250);  convolution_16 = unsqueeze_250 = None
    mul_297: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_256);  sub_76 = unsqueeze_256 = None
    sub_77: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_297);  where_16 = mul_297 = None
    sub_78: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_253);  sub_77 = unsqueeze_253 = None
    mul_298: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_259);  sub_78 = unsqueeze_259 = None
    mul_299: "f32[160]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_28);  sum_32 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_298, convolution_15, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_298 = convolution_15 = primals_64 = None
    getitem_124: "f32[8, 160, 28, 28]" = convolution_backward_24[0]
    getitem_125: "f32[160, 160, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(getitem_124, relu_8, primals_63, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False]);  getitem_124 = primals_63 = None
    getitem_127: "f32[8, 160, 28, 28]" = convolution_backward_25[0]
    getitem_128: "f32[160, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_66: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_67: "f32[8, 160, 28, 28]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_14: "b8[8, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 160, 28, 28]" = torch.ops.aten.where.self(le_14, scalar_tensor_17, getitem_127);  le_14 = scalar_tensor_17 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_261: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_33: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_79: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_262)
    mul_300: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_79);  sub_79 = None
    sum_34: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
    mul_301: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
    unsqueeze_263: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_264: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_302: "f32[160]" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
    mul_303: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_304: "f32[160]" = torch.ops.aten.mul.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    unsqueeze_266: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_267: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_305: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_269: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
    unsqueeze_270: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_80: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_262);  convolution_14 = unsqueeze_262 = None
    mul_306: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_268);  sub_80 = unsqueeze_268 = None
    sub_81: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_306);  where_17 = mul_306 = None
    sub_82: "f32[8, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_265);  sub_81 = unsqueeze_265 = None
    mul_307: "f32[8, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_271);  sub_82 = unsqueeze_271 = None
    mul_308: "f32[160]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_25);  sum_34 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_307, getitem_16, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = getitem_16 = primals_62 = None
    getitem_130: "f32[8, 256, 28, 28]" = convolution_backward_26[0]
    getitem_131: "f32[160, 256, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_130: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_9, getitem_130);  slice_9 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_2: "f32[8, 256, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_130, mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_17);  add_130 = mul_56 = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    mul_309: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, relu_7)
    mul_310: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward_2, div);  max_pool2d_with_indices_backward_2 = div = None
    sum_35: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2, 3], True);  mul_309 = None
    gt_3: "b8[8, 256, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_13, -3.0)
    lt_3: "b8[8, 256, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_13, 3.0);  convolution_13 = None
    bitwise_and_3: "b8[8, 256, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_3);  gt_3 = lt_3 = None
    mul_311: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_35, 0.16666666666666666);  sum_35 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[8, 256, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_311, scalar_tensor_18);  bitwise_and_3 = mul_311 = scalar_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_18, mean, primals_60, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_18 = mean = primals_60 = None
    getitem_133: "f32[8, 256, 1, 1]" = convolution_backward_27[0]
    getitem_134: "f32[256, 256, 1, 1]" = convolution_backward_27[1]
    getitem_135: "f32[256]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 256, 56, 56]" = torch.ops.aten.expand.default(getitem_133, [8, 256, 56, 56]);  getitem_133 = None
    div_8: "f32[8, 256, 56, 56]" = torch.ops.aten.div.Scalar(expand_4, 3136);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    add_131: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_310, div_8);  mul_310 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_69: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_70: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_15: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_15, scalar_tensor_19, add_131);  le_15 = scalar_tensor_19 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_273: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_83: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_274)
    mul_312: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_83);  sub_83 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3]);  mul_312 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
    unsqueeze_275: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
    unsqueeze_276: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_316: "f32[256]" = torch.ops.aten.mul.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    unsqueeze_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_281: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_282: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_84: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_274);  convolution_12 = unsqueeze_274 = None
    mul_318: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_280);  sub_84 = unsqueeze_280 = None
    sub_85: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_318);  where_19 = mul_318 = None
    sub_86: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_277);  sub_85 = unsqueeze_277 = None
    mul_319: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_283);  sub_86 = unsqueeze_283 = None
    mul_320: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_22);  sum_37 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_319, cat, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_319 = cat = primals_59 = None
    getitem_136: "f32[8, 448, 56, 56]" = convolution_backward_28[0]
    getitem_137: "f32[256, 448, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_13: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 0, 64)
    slice_14: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 64, 192)
    slice_15: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 192, 320)
    slice_16: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_136, 1, 320, 448);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_73: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_16: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_16, scalar_tensor_20, slice_16);  le_16 = scalar_tensor_20 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_285: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_38: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_87: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_286)
    mul_321: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_20, sub_87);  sub_87 = None
    sum_39: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3]);  mul_321 = None
    mul_322: "f32[128]" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
    unsqueeze_287: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_288: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_323: "f32[128]" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
    mul_324: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_325: "f32[128]" = torch.ops.aten.mul.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    unsqueeze_290: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
    unsqueeze_291: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_326: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_293: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_294: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_88: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_286);  convolution_11 = unsqueeze_286 = None
    mul_327: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_292);  sub_88 = unsqueeze_292 = None
    sub_89: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_20, mul_327);  where_20 = mul_327 = None
    sub_90: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_289);  sub_89 = unsqueeze_289 = None
    mul_328: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_295);  sub_90 = unsqueeze_295 = None
    mul_329: "f32[128]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_19);  sum_39 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_328, convolution_10, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = convolution_10 = primals_58 = None
    getitem_139: "f32[8, 128, 56, 56]" = convolution_backward_29[0]
    getitem_140: "f32[128, 128, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(getitem_139, relu_5, primals_57, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_139 = primals_57 = None
    getitem_142: "f32[8, 128, 56, 56]" = convolution_backward_30[0]
    getitem_143: "f32[128, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_132: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_15, getitem_142);  slice_15 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_75: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_76: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_17: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_17, scalar_tensor_21, add_132);  le_17 = scalar_tensor_21 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_297: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_40: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_91: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_298)
    mul_330: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_21, sub_91);  sub_91 = None
    sum_41: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
    mul_331: "f32[128]" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
    unsqueeze_299: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_300: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_332: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
    mul_333: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_334: "f32[128]" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_302: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_303: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_335: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_305: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_306: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_92: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_298);  convolution_9 = unsqueeze_298 = None
    mul_336: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_304);  sub_92 = unsqueeze_304 = None
    sub_93: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_21, mul_336);  where_21 = mul_336 = None
    sub_94: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_301);  sub_93 = unsqueeze_301 = None
    mul_337: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_307);  sub_94 = unsqueeze_307 = None
    mul_338: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_16);  sum_41 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_337, convolution_8, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = convolution_8 = primals_56 = None
    getitem_145: "f32[8, 128, 56, 56]" = convolution_backward_31[0]
    getitem_146: "f32[128, 128, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(getitem_145, relu_4, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_145 = primals_55 = None
    getitem_148: "f32[8, 128, 56, 56]" = convolution_backward_32[0]
    getitem_149: "f32[128, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    add_133: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_14, getitem_148);  slice_14 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_78: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_79: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_18: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_18, scalar_tensor_22, add_133);  le_18 = scalar_tensor_22 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_42: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_95: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_310)
    mul_339: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_22, sub_95);  sub_95 = None
    sum_43: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 2, 3]);  mul_339 = None
    mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
    unsqueeze_311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
    unsqueeze_312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_341: "f32[128]" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
    mul_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_343: "f32[128]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
    unsqueeze_314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_344: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
    unsqueeze_318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_96: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_310);  convolution_7 = unsqueeze_310 = None
    mul_345: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_316);  sub_96 = unsqueeze_316 = None
    sub_97: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_22, mul_345);  where_22 = mul_345 = None
    sub_98: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_313);  sub_97 = unsqueeze_313 = None
    mul_346: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_319);  sub_98 = unsqueeze_319 = None
    mul_347: "f32[128]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_13);  sum_43 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_346, convolution_6, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_346 = convolution_6 = primals_54 = None
    getitem_151: "f32[8, 128, 56, 56]" = convolution_backward_33[0]
    getitem_152: "f32[128, 128, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_151, relu_3, primals_53, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  getitem_151 = primals_53 = None
    getitem_154: "f32[8, 128, 56, 56]" = convolution_backward_34[0]
    getitem_155: "f32[128, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_81: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_82: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_19: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_19, scalar_tensor_23, getitem_154);  le_19 = scalar_tensor_23 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_99: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_322)
    mul_348: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_23, sub_99);  sub_99 = None
    sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
    unsqueeze_323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
    unsqueeze_324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_350: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
    unsqueeze_327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_353: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_100: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_322);  convolution_5 = unsqueeze_322 = None
    mul_354: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_328);  sub_100 = unsqueeze_328 = None
    sub_101: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_23, mul_354);  where_23 = mul_354 = None
    sub_102: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_325);  sub_101 = unsqueeze_325 = None
    mul_355: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_331);  sub_102 = unsqueeze_331 = None
    mul_356: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_10);  sum_45 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_355, relu_2, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_355 = primals_52 = None
    getitem_157: "f32[8, 64, 56, 56]" = convolution_backward_35[0]
    getitem_158: "f32[128, 64, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_134: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_13, getitem_157);  slice_13 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_84: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_85: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_20: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_20, scalar_tensor_24, add_134);  le_20 = scalar_tensor_24 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_333: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_46: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_103: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_334)
    mul_357: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_24, sub_103);  sub_103 = None
    sum_47: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_358: "f32[64]" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
    unsqueeze_335: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_336: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_359: "f32[64]" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
    mul_360: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_361: "f32[64]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    unsqueeze_338: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_339: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_362: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_341: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_342: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_104: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_334);  convolution_4 = unsqueeze_334 = None
    mul_363: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_340);  sub_104 = unsqueeze_340 = None
    sub_105: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_24, mul_363);  where_24 = mul_363 = None
    sub_106: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_337);  sub_105 = unsqueeze_337 = None
    mul_364: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_343);  sub_106 = unsqueeze_343 = None
    mul_365: "f32[64]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_7);  sum_47 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_364, convolution_3, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = convolution_3 = primals_51 = None
    getitem_160: "f32[8, 64, 56, 56]" = convolution_backward_36[0]
    getitem_161: "f32[64, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(getitem_160, relu_1, primals_50, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_160 = primals_50 = None
    getitem_163: "f32[8, 64, 112, 112]" = convolution_backward_37[0]
    getitem_164: "f32[64, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_87: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_88: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_21: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_21, scalar_tensor_25, getitem_163);  le_21 = scalar_tensor_25 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_345: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_48: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_107: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_346)
    mul_366: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_25, sub_107);  sub_107 = None
    sum_49: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
    mul_367: "f32[64]" = torch.ops.aten.mul.Tensor(sum_48, 9.964923469387754e-06)
    unsqueeze_347: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
    unsqueeze_348: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_368: "f32[64]" = torch.ops.aten.mul.Tensor(sum_49, 9.964923469387754e-06)
    mul_369: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_370: "f32[64]" = torch.ops.aten.mul.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_350: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_351: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_371: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_353: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_354: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_108: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_346);  convolution_2 = unsqueeze_346 = None
    mul_372: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_352);  sub_108 = unsqueeze_352 = None
    sub_109: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_25, mul_372);  where_25 = mul_372 = None
    sub_110: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_349);  sub_109 = unsqueeze_349 = None
    mul_373: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_355);  sub_110 = unsqueeze_355 = None
    mul_374: "f32[64]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_4);  sum_49 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_373, convolution_1, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = convolution_1 = primals_49 = None
    getitem_166: "f32[8, 64, 112, 112]" = convolution_backward_38[0]
    getitem_167: "f32[64, 64, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(getitem_166, relu, primals_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  getitem_166 = primals_48 = None
    getitem_169: "f32[8, 64, 112, 112]" = convolution_backward_39[0]
    getitem_170: "f32[64, 1, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_90: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_91: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_22: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_22, scalar_tensor_26, getitem_169);  le_22 = scalar_tensor_26 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_50: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_111: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_358)
    mul_375: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_26, sub_111);  sub_111 = None
    sum_51: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
    mul_376: "f32[64]" = torch.ops.aten.mul.Tensor(sum_50, 9.964923469387754e-06)
    unsqueeze_359: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_360: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_377: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, 9.964923469387754e-06)
    mul_378: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_379: "f32[64]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    unsqueeze_362: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_363: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_380: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_365: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_366: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_112: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_358);  convolution = unsqueeze_358 = None
    mul_381: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_364);  sub_112 = unsqueeze_364 = None
    sub_113: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_26, mul_381);  where_26 = mul_381 = None
    sub_114: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_361);  sub_113 = unsqueeze_361 = None
    mul_382: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_367);  sub_114 = unsqueeze_367 = None
    mul_383: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_1);  sum_51 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_382, primals_163, primals_47, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_382 = primals_163 = primals_47 = None
    getitem_173: "f32[64, 3, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
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
    return pytree.tree_unflatten([addmm, mul_383, sum_50, mul_374, sum_48, mul_365, sum_46, mul_356, sum_44, mul_347, sum_42, mul_338, sum_40, mul_329, sum_38, mul_320, sum_36, mul_308, sum_33, mul_299, sum_31, mul_290, sum_29, mul_281, sum_27, mul_272, sum_25, mul_260, sum_22, mul_251, sum_20, mul_242, sum_18, mul_233, sum_16, mul_224, sum_14, mul_212, sum_11, mul_203, sum_9, mul_194, sum_7, mul_185, sum_5, mul_176, sum_3, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_135, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_108, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, getitem_81, getitem_77, getitem_74, getitem_71, getitem_68, getitem_65, getitem_62, getitem_59, getitem_56, getitem_53, getitem_54, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    