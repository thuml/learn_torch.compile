from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[24]"; primals_2: "f32[24]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64]"; primals_11: "f32[256]"; primals_12: "f32[256]"; primals_13: "f32[256]"; primals_14: "f32[256]"; primals_15: "f32[64]"; primals_16: "f32[64]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[256]"; primals_20: "f32[256]"; primals_21: "f32[128]"; primals_22: "f32[128]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128]"; primals_32: "f32[128]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[256]"; primals_38: "f32[256]"; primals_39: "f32[1024]"; primals_40: "f32[1024]"; primals_41: "f32[1024]"; primals_42: "f32[1024]"; primals_43: "f32[256]"; primals_44: "f32[256]"; primals_45: "f32[31, 64]"; primals_46: "f32[31, 64]"; primals_47: "f32[256]"; primals_48: "f32[256]"; primals_49: "f32[1024]"; primals_50: "f32[1024]"; primals_51: "f32[512]"; primals_52: "f32[512]"; primals_53: "f32[31, 128]"; primals_54: "f32[31, 128]"; primals_55: "f32[512]"; primals_56: "f32[512]"; primals_57: "f32[2048]"; primals_58: "f32[2048]"; primals_59: "f32[2048]"; primals_60: "f32[2048]"; primals_61: "f32[512]"; primals_62: "f32[512]"; primals_63: "f32[15, 128]"; primals_64: "f32[15, 128]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[2048]"; primals_68: "f32[2048]"; primals_69: "f32[24, 3, 3, 3]"; primals_70: "f32[32, 24, 3, 3]"; primals_71: "f32[64, 32, 3, 3]"; primals_72: "f32[64, 64, 1, 1]"; primals_73: "f32[64, 64, 3, 3]"; primals_74: "f32[256, 64, 1, 1]"; primals_75: "f32[256, 64, 1, 1]"; primals_76: "f32[64, 256, 1, 1]"; primals_77: "f32[64, 64, 3, 3]"; primals_78: "f32[256, 64, 1, 1]"; primals_79: "f32[128, 256, 1, 1]"; primals_80: "f32[128, 128, 3, 3]"; primals_81: "f32[512, 128, 1, 1]"; primals_82: "f32[512, 256, 1, 1]"; primals_83: "f32[128, 512, 1, 1]"; primals_84: "f32[128, 128, 3, 3]"; primals_85: "f32[512, 128, 1, 1]"; primals_86: "f32[256, 512, 1, 1]"; primals_87: "f32[256, 256, 3, 3]"; primals_88: "f32[1024, 256, 1, 1]"; primals_89: "f32[1024, 512, 1, 1]"; primals_90: "f32[256, 1024, 1, 1]"; primals_91: "f32[768, 256, 1, 1]"; primals_92: "f32[1024, 256, 1, 1]"; primals_93: "f32[512, 1024, 1, 1]"; primals_94: "f32[1536, 512, 1, 1]"; primals_95: "f32[2048, 512, 1, 1]"; primals_96: "f32[2048, 1024, 1, 1]"; primals_97: "f32[512, 2048, 1, 1]"; primals_98: "f32[1536, 512, 1, 1]"; primals_99: "f32[2048, 512, 1, 1]"; primals_100: "f32[1000, 2048]"; primals_101: "f32[1000]"; primals_102: "i64[]"; primals_103: "f32[24]"; primals_104: "f32[24]"; primals_105: "i64[]"; primals_106: "f32[32]"; primals_107: "f32[32]"; primals_108: "i64[]"; primals_109: "f32[64]"; primals_110: "f32[64]"; primals_111: "i64[]"; primals_112: "f32[64]"; primals_113: "f32[64]"; primals_114: "i64[]"; primals_115: "f32[64]"; primals_116: "f32[64]"; primals_117: "i64[]"; primals_118: "f32[256]"; primals_119: "f32[256]"; primals_120: "i64[]"; primals_121: "f32[256]"; primals_122: "f32[256]"; primals_123: "i64[]"; primals_124: "f32[64]"; primals_125: "f32[64]"; primals_126: "i64[]"; primals_127: "f32[64]"; primals_128: "f32[64]"; primals_129: "i64[]"; primals_130: "f32[256]"; primals_131: "f32[256]"; primals_132: "i64[]"; primals_133: "f32[128]"; primals_134: "f32[128]"; primals_135: "i64[]"; primals_136: "f32[128]"; primals_137: "f32[128]"; primals_138: "i64[]"; primals_139: "f32[512]"; primals_140: "f32[512]"; primals_141: "i64[]"; primals_142: "f32[512]"; primals_143: "f32[512]"; primals_144: "i64[]"; primals_145: "f32[128]"; primals_146: "f32[128]"; primals_147: "i64[]"; primals_148: "f32[128]"; primals_149: "f32[128]"; primals_150: "i64[]"; primals_151: "f32[512]"; primals_152: "f32[512]"; primals_153: "i64[]"; primals_154: "f32[256]"; primals_155: "f32[256]"; primals_156: "i64[]"; primals_157: "f32[256]"; primals_158: "f32[256]"; primals_159: "i64[]"; primals_160: "f32[1024]"; primals_161: "f32[1024]"; primals_162: "i64[]"; primals_163: "f32[1024]"; primals_164: "f32[1024]"; primals_165: "i64[]"; primals_166: "f32[256]"; primals_167: "f32[256]"; primals_168: "i64[]"; primals_169: "f32[256]"; primals_170: "f32[256]"; primals_171: "i64[]"; primals_172: "f32[1024]"; primals_173: "f32[1024]"; primals_174: "i64[]"; primals_175: "f32[512]"; primals_176: "f32[512]"; primals_177: "i64[]"; primals_178: "f32[512]"; primals_179: "f32[512]"; primals_180: "i64[]"; primals_181: "f32[2048]"; primals_182: "f32[2048]"; primals_183: "i64[]"; primals_184: "f32[2048]"; primals_185: "f32[2048]"; primals_186: "i64[]"; primals_187: "f32[512]"; primals_188: "f32[512]"; primals_189: "i64[]"; primals_190: "f32[512]"; primals_191: "f32[512]"; primals_192: "i64[]"; primals_193: "f32[2048]"; primals_194: "f32[2048]"; primals_195: "f32[8, 3, 256, 256]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(primals_195, primals_69, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_102, 1)
    
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
    mul_2: "f32[24]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[24]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[24]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
    add_3: "f32[24]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 24, 128, 128]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(relu, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 128, 128]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(relu_1, primals_71, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_108, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [3, 3], [2, 2], [1, 1])
    getitem_6: "f32[8, 64, 64, 64]" = max_pool2d_with_indices[0]
    getitem_7: "i64[8, 64, 64, 64]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_21: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_3, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_114, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_28: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[64]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000030518509476);  squeeze_14 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_4, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
    mul_35: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[256]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_120, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
    mul_42: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_35: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_29, add_34);  add_29 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_5: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_5, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
    mul_49: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[64]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
    add_39: "f32[64]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_6, primals_77, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_126, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
    mul_56: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
    add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_7, primals_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 256, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 256, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
    mul_63: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
    add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[256]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
    add_49: "f32[256]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_51: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_50, relu_5);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_8: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_8, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_132, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
    mul_70: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_73: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_74: "f32[128]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
    add_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_9, primals_80, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1, 1]" = var_mean_11[0]
    getitem_25: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
    mul_77: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[128]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001220852154804);  squeeze_35 = None
    mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_10, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_138, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1, 1]" = var_mean_12[0]
    getitem_27: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
    mul_84: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[512]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_87: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001220852154804);  squeeze_38 = None
    mul_88: "f32[512]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
    add_65: "f32[512]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_8, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1, 1]" = var_mean_13[0]
    getitem_29: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
    mul_91: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
    add_69: "f32[512]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_94: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
    mul_95: "f32[512]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
    add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_72: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_66, add_71);  add_66 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_11: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_11, primals_83, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_144, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_31: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
    mul_98: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[128]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_101: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_102: "f32[128]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[128]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
    add_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_12, primals_84, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_33: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
    mul_105: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
    add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_82);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_13, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_150, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1, 1]" = var_mean_16[0]
    getitem_35: "f32[1, 512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_16: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
    mul_112: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_49: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[512]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
    add_85: "f32[512]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_115: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_116: "f32[512]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[512]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
    add_86: "f32[512]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_88: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_87, relu_11);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_14: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_14, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 256, 1, 1]" = var_mean_17[0]
    getitem_37: "f32[1, 256, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_17: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
    mul_119: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_52: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
    add_91: "f32[256]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_122: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
    add_92: "f32[256]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_15, primals_87, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_156, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 256, 1, 1]" = var_mean_18[0]
    getitem_39: "f32[1, 256, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_18: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
    mul_126: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_55: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[256]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
    add_96: "f32[256]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0004885197850513);  squeeze_56 = None
    mul_130: "f32[256]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
    add_97: "f32[256]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_98);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_16, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1, 1]" = var_mean_19[0]
    getitem_41: "f32[1, 1024, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_19: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
    mul_133: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_58: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
    add_101: "f32[1024]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_136: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0004885197850513);  squeeze_59 = None
    mul_137: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
    add_102: "f32[1024]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_14, primals_89, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_162, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1, 1]" = var_mean_20[0]
    getitem_43: "f32[1, 1024, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_20: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
    mul_140: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_61: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
    add_106: "f32[1024]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_143: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
    mul_144: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
    add_107: "f32[1024]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_109: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_103, add_108);  add_103 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_17: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_17, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_21[0]
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_21: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
    mul_147: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_64: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[256]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_112: "f32[256]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_151: "f32[256]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[256]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_22: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(relu_18, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(convolution_22, [256, 256, 256], 1);  convolution_22 = None
    getitem_46: "f32[8, 256, 16, 16]" = split_with_sizes[0]
    getitem_47: "f32[8, 256, 16, 16]" = split_with_sizes[1]
    getitem_48: "f32[8, 256, 16, 16]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_46, memory_format = torch.contiguous_format);  getitem_46 = None
    view: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone, [32, 64, 256]);  clone = None
    permute: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_1: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_47, memory_format = torch.contiguous_format);  getitem_47 = None
    view_1: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_1, [32, 64, 256]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_2: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_48, memory_format = torch.contiguous_format);  getitem_48 = None
    view_2: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_2, [32, 64, 256]);  clone_2 = None
    permute_1: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute, [32, 256, 64])
    view_3: "f32[32, 256, 64]" = torch.ops.aten.view.default(expand, [32, 256, 64]);  expand = None
    expand_1: "f32[32, 64, 256]" = torch.ops.aten.expand.default(view_1, [32, 64, 256]);  view_1 = None
    view_4: "f32[32, 64, 256]" = torch.ops.aten.view.default(expand_1, [32, 64, 256]);  expand_1 = None
    bmm: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_3, view_4)
    view_5: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm, [32, 256, 256]);  bmm = None
    mul_154: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_5, 0.125);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_6: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(permute, [32, 16, 16, 64]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_2: "f32[64, 31]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_3: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(view_6, memory_format = torch.contiguous_format)
    view_7: "f32[8192, 64]" = torch.ops.aten.view.default(clone_3, [8192, 64]);  clone_3 = None
    mm: "f32[8192, 31]" = torch.ops.aten.mm.default(view_7, permute_2)
    view_8: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm, [32, 16, 16, 31]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_9: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_8, [-1, 16, 31]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_9, [0, 1], 0.0);  view_9 = None
    view_10: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd, [512, 512]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_1: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_10, [0, 15], 0.0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_11: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_1, [-1, 17, 31]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_1: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_11, 0, 0, 9223372036854775807);  view_11 = None
    slice_2: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 16);  slice_1 = None
    slice_3: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_2, 2, 15, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_12: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_3, [32, 16, 1, 16, 16]);  slice_3 = None
    expand_2: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_12, [-1, -1, 16, -1, -1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_3: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_2, [0, 1, 3, 2, 4]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_4: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_5: "f32[64, 31]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    clone_4: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_13: "f32[8192, 64]" = torch.ops.aten.view.default(clone_4, [8192, 64]);  clone_4 = None
    mm_1: "f32[8192, 31]" = torch.ops.aten.mm.default(view_13, permute_5)
    view_14: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_1, [32, 16, 16, 31]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_15: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_14, [-1, 16, 31]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_2: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_15, [0, 1], 0.0);  view_15 = None
    view_16: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_2, [512, 512]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_3: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_16, [0, 15], 0.0);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_17: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_3, [-1, 17, 31]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_4: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_17, 0, 0, 9223372036854775807);  view_17 = None
    slice_5: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 16);  slice_4 = None
    slice_6: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_5, 2, 15, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_18: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_6, [32, 16, 1, 16, 16]);  slice_6 = None
    expand_3: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_18, [-1, -1, 16, -1, -1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_6: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_3, [0, 3, 1, 4, 2]);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_115: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_6, permute_3);  permute_6 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_5: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format);  add_115 = None
    view_19: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_5, [32, 256, 256]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_116: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_154, view_19);  mul_154 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
    sub_22: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_116, amax);  add_116 = amax = None
    exp: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_1: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_19: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_4: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div, [32, 256, 256]);  div = None
    view_20: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_4, [32, 256, 256]);  expand_4 = None
    expand_5: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_1, [32, 256, 64]);  permute_1 = None
    view_21: "f32[32, 256, 64]" = torch.ops.aten.view.default(expand_5, [32, 256, 64]);  expand_5 = None
    bmm_1: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_1, [32, 256, 64]);  bmm_1 = None
    permute_7: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    clone_6: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_23: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_6, [8, 256, 16, 16]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_168, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_49: "f32[1, 256, 1, 1]" = var_mean_22[0]
    getitem_50: "f32[1, 256, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05)
    rsqrt_22: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_23, getitem_50)
    mul_155: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
    squeeze_66: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    squeeze_67: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_156: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_157: "f32[256]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
    add_119: "f32[256]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_68: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    mul_158: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_159: "f32[256]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[256]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
    add_120: "f32[256]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_161: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_89);  mul_155 = unsqueeze_89 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_91);  mul_161 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_19, primals_92, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_51: "f32[1, 1024, 1, 1]" = var_mean_23[0]
    getitem_52: "f32[1, 1024, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-05)
    rsqrt_23: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_52)
    mul_162: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
    squeeze_69: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    squeeze_70: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_163: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_164: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
    add_124: "f32[1024]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    squeeze_71: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    mul_165: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_166: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_165, 0.1);  mul_165 = None
    mul_167: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
    add_125: "f32[1024]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    unsqueeze_92: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_93: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_168: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_93);  mul_162 = unsqueeze_93 = None
    unsqueeze_94: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_95: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_95);  mul_168 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_127: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_126, relu_17);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    relu_20: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_127);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_20, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_174, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_53: "f32[1, 512, 1, 1]" = var_mean_24[0]
    getitem_54: "f32[1, 512, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05)
    rsqrt_24: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_54)
    mul_169: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    squeeze_72: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    squeeze_73: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_170: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_171: "f32[512]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_130: "f32[512]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    squeeze_74: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    mul_172: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_173: "f32[512]" = torch.ops.aten.mul.Tensor(mul_172, 0.1);  mul_172 = None
    mul_174: "f32[512]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
    add_131: "f32[512]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_175: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_97);  mul_169 = unsqueeze_97 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_99);  mul_175 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_25: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(relu_21, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(convolution_25, [512, 512, 512], 1);  convolution_25 = None
    getitem_55: "f32[8, 512, 16, 16]" = split_with_sizes_1[0]
    getitem_56: "f32[8, 512, 16, 16]" = split_with_sizes_1[1]
    getitem_57: "f32[8, 512, 16, 16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_7: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_55, memory_format = torch.contiguous_format);  getitem_55 = None
    view_24: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_7, [32, 128, 256]);  clone_7 = None
    permute_8: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_8: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_56, memory_format = torch.contiguous_format);  getitem_56 = None
    view_25: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_8, [32, 128, 256]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_9: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    view_26: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_9, [32, 128, 256]);  clone_9 = None
    permute_9: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_6: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_8, [32, 256, 128])
    view_27: "f32[32, 256, 128]" = torch.ops.aten.view.default(expand_6, [32, 256, 128]);  expand_6 = None
    expand_7: "f32[32, 128, 256]" = torch.ops.aten.expand.default(view_25, [32, 128, 256]);  view_25 = None
    view_28: "f32[32, 128, 256]" = torch.ops.aten.view.default(expand_7, [32, 128, 256]);  expand_7 = None
    bmm_2: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_27, view_28)
    view_29: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_2, [32, 256, 256]);  bmm_2 = None
    mul_176: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_29, 0.08838834764831845);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_30: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(permute_8, [32, 16, 16, 128]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_10: "f32[128, 31]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    clone_10: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(view_30, memory_format = torch.contiguous_format)
    view_31: "f32[8192, 128]" = torch.ops.aten.view.default(clone_10, [8192, 128]);  clone_10 = None
    mm_2: "f32[8192, 31]" = torch.ops.aten.mm.default(view_31, permute_10)
    view_32: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_2, [32, 16, 16, 31]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_33: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_32, [-1, 16, 31]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_4: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_33, [0, 1], 0.0);  view_33 = None
    view_34: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_4, [512, 512]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_5: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_34, [0, 15], 0.0);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_35: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_5, [-1, 17, 31]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_7: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_35, 0, 0, 9223372036854775807);  view_35 = None
    slice_8: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 16);  slice_7 = None
    slice_9: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_8, 2, 15, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_36: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_9, [32, 16, 1, 16, 16]);  slice_9 = None
    expand_8: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_36, [-1, -1, 16, -1, -1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_11: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_8, [0, 1, 3, 2, 4]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_12: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_13: "f32[128, 31]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    clone_11: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_37: "f32[8192, 128]" = torch.ops.aten.view.default(clone_11, [8192, 128]);  clone_11 = None
    mm_3: "f32[8192, 31]" = torch.ops.aten.mm.default(view_37, permute_13)
    view_38: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_3, [32, 16, 16, 31]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_39: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_38, [-1, 16, 31]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_6: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_39, [0, 1], 0.0);  view_39 = None
    view_40: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_6, [512, 512]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_7: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_40, [0, 15], 0.0);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_41: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_7, [-1, 17, 31]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_10: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_41, 0, 0, 9223372036854775807);  view_41 = None
    slice_11: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 16);  slice_10 = None
    slice_12: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_11, 2, 15, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_42: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_12, [32, 16, 1, 16, 16]);  slice_12 = None
    expand_9: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_42, [-1, -1, 16, -1, -1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_9, [0, 3, 1, 4, 2]);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_133: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_14, permute_11);  permute_14 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_12: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format);  add_133 = None
    view_43: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_12, [32, 256, 256]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_134: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_176, view_43);  mul_176 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_134, [-1], True)
    sub_26: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_134, amax_1);  add_134 = amax_1 = None
    exp_1: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_23: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_10: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_1, [32, 256, 256]);  div_1 = None
    view_44: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_10, [32, 256, 256]);  expand_10 = None
    expand_11: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_9, [32, 256, 128]);  permute_9 = None
    view_45: "f32[32, 256, 128]" = torch.ops.aten.view.default(expand_11, [32, 256, 128]);  expand_11 = None
    bmm_3: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_3, [32, 256, 128]);  bmm_3 = None
    permute_15: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    clone_13: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_47: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_13, [8, 512, 16, 16]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_47, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(avg_pool2d, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1, 1]" = var_mean_25[0]
    getitem_59: "f32[1, 512, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_136: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_25: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_59)
    mul_177: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_25);  sub_27 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_178: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_179: "f32[512]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_137: "f32[512]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_180: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0019569471624266);  squeeze_77 = None
    mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(mul_180, 0.1);  mul_180 = None
    mul_182: "f32[512]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
    add_138: "f32[512]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_183: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_101);  mul_177 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_139: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_103);  mul_183 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_22, primals_95, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_180, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 2048, 1, 1]" = var_mean_26[0]
    getitem_61: "f32[1, 2048, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_141: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_26: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_61)
    mul_184: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_26);  sub_28 = None
    squeeze_78: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_79: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_185: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_186: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_142: "f32[2048]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_80: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_187: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0019569471624266);  squeeze_80 = None
    mul_188: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
    add_143: "f32[2048]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_104: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_105: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_190: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_105);  mul_184 = unsqueeze_105 = None
    unsqueeze_106: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_107: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_144: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_107);  mul_190 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_20, primals_96, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 2048, 1, 1]" = var_mean_27[0]
    getitem_63: "f32[1, 2048, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_146: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_27: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_29: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_63)
    mul_191: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_27);  sub_29 = None
    squeeze_81: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_82: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_192: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_193: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_147: "f32[2048]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    squeeze_83: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_194: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0019569471624266);  squeeze_83 = None
    mul_195: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_194, 0.1);  mul_194 = None
    mul_196: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
    add_148: "f32[2048]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    unsqueeze_108: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_109: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_197: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_109);  mul_191 = unsqueeze_109 = None
    unsqueeze_110: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_111: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_149: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_111);  mul_197 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_150: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_144, add_149);  add_144 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    relu_23: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_23, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_186, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512, 1, 1]" = var_mean_28[0]
    getitem_65: "f32[1, 512, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_28: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_65)
    mul_198: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
    squeeze_84: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_85: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_199: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_200: "f32[512]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_153: "f32[512]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    squeeze_86: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_201: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0019569471624266);  squeeze_86 = None
    mul_202: "f32[512]" = torch.ops.aten.mul.Tensor(mul_201, 0.1);  mul_201 = None
    mul_203: "f32[512]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_204: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_113);  mul_198 = unsqueeze_113 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_115);  mul_204 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_29: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(relu_24, primals_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(convolution_29, [512, 512, 512], 1);  convolution_29 = None
    getitem_66: "f32[8, 512, 8, 8]" = split_with_sizes_2[0]
    getitem_67: "f32[8, 512, 8, 8]" = split_with_sizes_2[1]
    getitem_68: "f32[8, 512, 8, 8]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_14: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_66, memory_format = torch.contiguous_format);  getitem_66 = None
    view_48: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_14, [32, 128, 64]);  clone_14 = None
    permute_16: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_15: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_67, memory_format = torch.contiguous_format);  getitem_67 = None
    view_49: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_15, [32, 128, 64]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_16: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_68, memory_format = torch.contiguous_format);  getitem_68 = None
    view_50: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_16, [32, 128, 64]);  clone_16 = None
    permute_17: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_12: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_16, [32, 64, 128])
    view_51: "f32[32, 64, 128]" = torch.ops.aten.view.default(expand_12, [32, 64, 128]);  expand_12 = None
    expand_13: "f32[32, 128, 64]" = torch.ops.aten.expand.default(view_49, [32, 128, 64]);  view_49 = None
    view_52: "f32[32, 128, 64]" = torch.ops.aten.view.default(expand_13, [32, 128, 64]);  expand_13 = None
    bmm_4: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_51, view_52)
    view_53: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_4, [32, 64, 64]);  bmm_4 = None
    mul_205: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_53, 0.08838834764831845);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_54: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(permute_16, [32, 8, 8, 128]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[128, 15]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    clone_17: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(view_54, memory_format = torch.contiguous_format)
    view_55: "f32[2048, 128]" = torch.ops.aten.view.default(clone_17, [2048, 128]);  clone_17 = None
    mm_4: "f32[2048, 15]" = torch.ops.aten.mm.default(view_55, permute_18)
    view_56: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_4, [32, 8, 8, 15]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_57: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_56, [-1, 8, 15]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 1], 0.0);  view_57 = None
    view_58: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_8, [256, 128]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_9: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_58, [0, 7], 0.0);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_59: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_9, [-1, 9, 15]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_13: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_59, 0, 0, 9223372036854775807);  view_59 = None
    slice_14: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 8);  slice_13 = None
    slice_15: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_14, 2, 7, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_60: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_15, [32, 8, 1, 8, 8]);  slice_15 = None
    expand_14: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_60, [-1, -1, 8, -1, -1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_14, [0, 1, 3, 2, 4]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_20: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_21: "f32[128, 15]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    clone_18: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_61: "f32[2048, 128]" = torch.ops.aten.view.default(clone_18, [2048, 128]);  clone_18 = None
    mm_5: "f32[2048, 15]" = torch.ops.aten.mm.default(view_61, permute_21)
    view_62: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_5, [32, 8, 8, 15]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_63: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_62, [-1, 8, 15]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_10: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_63, [0, 1], 0.0);  view_63 = None
    view_64: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_10, [256, 128]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_11: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_64, [0, 7], 0.0);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_65: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_11, [-1, 9, 15]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_16: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_65, 0, 0, 9223372036854775807);  view_65 = None
    slice_17: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 8);  slice_16 = None
    slice_18: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_17, 2, 7, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_66: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_18, [32, 8, 1, 8, 8]);  slice_18 = None
    expand_15: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_66, [-1, -1, 8, -1, -1]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_22: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_15, [0, 3, 1, 4, 2]);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_156: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_22, permute_19);  permute_22 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_156, memory_format = torch.contiguous_format);  add_156 = None
    view_67: "f32[32, 64, 64]" = torch.ops.aten.view.default(clone_19, [32, 64, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_157: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_205, view_67);  mul_205 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
    sub_31: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_157, amax_2);  add_157 = amax_2 = None
    exp_2: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_3: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_27: "f32[32, 64, 64]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_16: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_2, [32, 64, 64]);  div_2 = None
    view_68: "f32[32, 64, 64]" = torch.ops.aten.view.default(expand_16, [32, 64, 64]);  expand_16 = None
    expand_17: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_17, [32, 64, 128]);  permute_17 = None
    view_69: "f32[32, 64, 128]" = torch.ops.aten.view.default(expand_17, [32, 64, 128]);  expand_17 = None
    bmm_5: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(view_68, view_69)
    view_70: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_5, [32, 64, 128]);  bmm_5 = None
    permute_23: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    clone_20: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_71: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_20, [8, 512, 8, 8]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_69: "f32[1, 512, 1, 1]" = var_mean_29[0]
    getitem_70: "f32[1, 512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_159: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_29: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_32: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_71, getitem_70)
    mul_206: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_29);  sub_32 = None
    squeeze_87: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    squeeze_88: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_160: "f32[512]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    squeeze_89: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    mul_209: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
    mul_210: "f32[512]" = torch.ops.aten.mul.Tensor(mul_209, 0.1);  mul_209 = None
    mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_161: "f32[512]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_212: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_117);  mul_206 = unsqueeze_117 = None
    unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_162: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_119);  mul_212 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_162);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_25, primals_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_192, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_71: "f32[1, 2048, 1, 1]" = var_mean_30[0]
    getitem_72: "f32[1, 2048, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_164: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_30: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_33: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_72)
    mul_213: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_30);  sub_33 = None
    squeeze_90: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    squeeze_91: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_214: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_215: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_165: "f32[2048]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    squeeze_92: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_216: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
    mul_217: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
    mul_218: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_166: "f32[2048]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    unsqueeze_120: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_121: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_219: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_121);  mul_213 = unsqueeze_121 = None
    unsqueeze_122: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_123: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_167: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_123);  mul_219 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_168: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_167, relu_23);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    relu_26: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_168);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_72: "f32[8, 2048]" = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_21: "f32[8, 2048]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_24: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_101, clone_21, permute_24);  primals_101 = None
    permute_25: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_6: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
    permute_26: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_7: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_26, clone_21);  permute_26 = clone_21 = None
    permute_27: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_4: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_73: "f32[1000]" = torch.ops.aten.view.default(sum_4, [1000]);  sum_4 = None
    permute_28: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_74: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_18: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_74, [8, 2048, 8, 8]);  view_74 = None
    div_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand_18, 64);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    alias_31: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_32: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    le: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(alias_32, 0);  alias_32 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le, scalar_tensor, div_3);  le = scalar_tensor = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_124: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_125: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
    unsqueeze_126: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
    sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_34: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_126)
    mul_220: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_34);  sub_34 = None
    sum_6: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 2, 3]);  mul_220 = None
    mul_221: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    unsqueeze_127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
    unsqueeze_128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    mul_223: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_224: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_224, 0);  mul_224 = None
    unsqueeze_131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_225: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_67);  primals_67 = None
    unsqueeze_133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    sub_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_126);  convolution_30 = unsqueeze_126 = None
    mul_226: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_132);  sub_35 = unsqueeze_132 = None
    sub_36: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_226);  mul_226 = None
    sub_37: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_36, unsqueeze_129);  sub_36 = unsqueeze_129 = None
    mul_227: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_135);  sub_37 = unsqueeze_135 = None
    mul_228: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_227, relu_25, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_227 = primals_99 = None
    getitem_73: "f32[8, 512, 8, 8]" = convolution_backward[0]
    getitem_74: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_34: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_35: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_1: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_73);  le_1 = scalar_tensor_1 = getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_137: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
    unsqueeze_138: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_38: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_138)
    mul_229: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_38);  sub_38 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 2, 3]);  mul_229 = None
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_140: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    unsqueeze_142: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_233, 0);  mul_233 = None
    unsqueeze_143: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_234: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_65);  primals_65 = None
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
    unsqueeze_146: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    sub_39: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_138);  view_71 = unsqueeze_138 = None
    mul_235: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_144);  sub_39 = unsqueeze_144 = None
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_235);  where_1 = mul_235 = None
    sub_41: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_40, unsqueeze_141);  sub_40 = unsqueeze_141 = None
    mul_236: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_147);  sub_41 = unsqueeze_147 = None
    mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_75: "f32[32, 128, 64]" = torch.ops.aten.view.default(mul_236, [32, 128, 64]);  mul_236 = None
    permute_29: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    view_76: "f32[32, 64, 128]" = torch.ops.aten.view.default(permute_29, [32, 64, 128]);  permute_29 = None
    permute_30: "f32[32, 64, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    bmm_6: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(permute_30, view_76);  permute_30 = None
    permute_31: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    bmm_7: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_76, permute_31);  view_76 = permute_31 = None
    view_77: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_6, [32, 64, 128]);  bmm_6 = None
    view_78: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_7, [32, 64, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_36: "f32[32, 64, 64]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_238: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_78, alias_36);  view_78 = None
    sum_9: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [-1], True)
    mul_239: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(alias_36, sum_9);  alias_36 = sum_9 = None
    sub_42: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_79: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.view.default(sub_42, [32, 8, 8, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_32: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_79, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_10: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_32, [2], True);  permute_32 = None
    view_80: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_10, [256, 8, 8]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full, view_80, 2, 7, 9223372036854775807);  full = view_80 = None
    full_1: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 1, 0, 8);  full_1 = slice_scatter = None
    full_2: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_1, 0, 0, 9223372036854775807);  full_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_81: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_12: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_81, [0, -7]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_82: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_12, [256, 8, 16]);  constant_pad_nd_12 = None
    constant_pad_nd_13: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_82, [0, -1]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_83: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_13, [32, 8, 8, 15]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_84: "f32[2048, 15]" = torch.ops.aten.view.default(view_83, [2048, 15]);  view_83 = None
    permute_33: "f32[15, 2048]" = torch.ops.aten.permute.default(view_84, [1, 0])
    mm_8: "f32[15, 128]" = torch.ops.aten.mm.default(permute_33, view_61);  permute_33 = view_61 = None
    permute_34: "f32[128, 15]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    permute_35: "f32[15, 128]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_9: "f32[2048, 128]" = torch.ops.aten.mm.default(view_84, permute_35);  view_84 = permute_35 = None
    view_85: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_9, [32, 8, 8, 128]);  mm_9 = None
    permute_36: "f32[15, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_37: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_38: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_79, [0, 1, 3, 2, 4]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_11: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_38, [2], True);  permute_38 = None
    view_86: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_11, [256, 8, 8]);  sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_3: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_3: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_3, view_86, 2, 7, 9223372036854775807);  full_3 = view_86 = None
    full_4: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_3, 1, 0, 8);  full_4 = slice_scatter_3 = None
    full_5: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_4, 0, 0, 9223372036854775807);  full_5 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_87: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_14: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_87, [0, -7]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_88: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_14, [256, 8, 16]);  constant_pad_nd_14 = None
    constant_pad_nd_15: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_88, [0, -1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_89: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_15, [32, 8, 8, 15]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_90: "f32[2048, 15]" = torch.ops.aten.view.default(view_89, [2048, 15]);  view_89 = None
    permute_39: "f32[15, 2048]" = torch.ops.aten.permute.default(view_90, [1, 0])
    mm_10: "f32[15, 128]" = torch.ops.aten.mm.default(permute_39, view_55);  permute_39 = view_55 = None
    permute_40: "f32[128, 15]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    permute_41: "f32[15, 128]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_11: "f32[2048, 128]" = torch.ops.aten.mm.default(view_90, permute_41);  view_90 = permute_41 = None
    view_91: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_11, [32, 8, 8, 128]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_169: "f32[32, 8, 8, 128]" = torch.ops.aten.add.Tensor(permute_37, view_91);  permute_37 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_42: "f32[15, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_22: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(add_169, memory_format = torch.contiguous_format);  add_169 = None
    view_92: "f32[32, 64, 128]" = torch.ops.aten.view.default(clone_22, [32, 64, 128]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_240: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(sub_42, 0.08838834764831845);  sub_42 = None
    view_93: "f32[32, 64, 64]" = torch.ops.aten.view.default(mul_240, [32, 64, 64]);  mul_240 = None
    permute_43: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    bmm_8: "f32[32, 128, 64]" = torch.ops.aten.bmm.default(permute_43, view_93);  permute_43 = None
    permute_44: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_9: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(view_93, permute_44);  view_93 = permute_44 = None
    view_94: "f32[32, 128, 64]" = torch.ops.aten.view.default(bmm_8, [32, 128, 64]);  bmm_8 = None
    view_95: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_9, [32, 64, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_170: "f32[32, 64, 128]" = torch.ops.aten.add.Tensor(view_92, view_95);  view_92 = view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_45: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    clone_23: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_96: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_23, [8, 512, 8, 8]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_97: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(view_94, [8, 512, 8, 8]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_46: "f32[32, 128, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1]);  add_170 = None
    clone_24: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_98: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_24, [8, 512, 8, 8]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat: "f32[8, 1536, 8, 8]" = torch.ops.aten.cat.default([view_98, view_97, view_96], 1);  view_98 = view_97 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, relu_24, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = primals_98 = None
    getitem_76: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_77: "f32[1536, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_38: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_39: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_2: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_76);  le_2 = scalar_tensor_2 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_149: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
    unsqueeze_150: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_150)
    mul_241: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_43);  sub_43 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_242: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_151: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_242, 0);  mul_242 = None
    unsqueeze_152: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_245: "f32[512]" = torch.ops.aten.mul.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
    unsqueeze_155: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_157: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_158: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    sub_44: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_150);  convolution_28 = unsqueeze_150 = None
    mul_247: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_156);  sub_44 = unsqueeze_156 = None
    sub_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_247);  where_2 = mul_247 = None
    sub_46: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_153);  sub_45 = unsqueeze_153 = None
    mul_248: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_159);  sub_46 = unsqueeze_159 = None
    mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_248, relu_23, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_248 = primals_97 = None
    getitem_79: "f32[8, 2048, 8, 8]" = convolution_backward_2[0]
    getitem_80: "f32[512, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_171: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(where, getitem_79);  where = getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    alias_41: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_42: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_3: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(alias_42, 0);  alias_42 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add_171);  le_3 = scalar_tensor_3 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_161: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_47: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_162)
    mul_250: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_47);  sub_47 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 2, 3]);  mul_250 = None
    mul_251: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_251, 0);  mul_251 = None
    unsqueeze_164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_252: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_253: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_254: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_254, 0);  mul_254 = None
    unsqueeze_167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_255: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_255, 0);  mul_255 = None
    unsqueeze_170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    sub_48: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_162);  convolution_27 = unsqueeze_162 = None
    mul_256: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_168);  sub_48 = unsqueeze_168 = None
    sub_49: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_256);  mul_256 = None
    sub_50: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_165);  sub_49 = unsqueeze_165 = None
    mul_257: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_171);  sub_50 = unsqueeze_171 = None
    mul_258: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_257, relu_20, primals_96, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_257 = primals_96 = None
    getitem_82: "f32[8, 1024, 16, 16]" = convolution_backward_3[0]
    getitem_83: "f32[2048, 1024, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_172: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_173: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_51: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_174)
    mul_259: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_51);  sub_51 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 2, 3]);  mul_259 = None
    mul_260: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_175: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
    unsqueeze_176: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
    mul_261: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_262: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_263: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    unsqueeze_178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_263, 0);  mul_263 = None
    unsqueeze_179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_264: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_57);  primals_57 = None
    unsqueeze_181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_264, 0);  mul_264 = None
    unsqueeze_182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    sub_52: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_174);  convolution_26 = unsqueeze_174 = None
    mul_265: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_180);  sub_52 = unsqueeze_180 = None
    sub_53: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_265);  where_3 = mul_265 = None
    sub_54: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_177);  sub_53 = unsqueeze_177 = None
    mul_266: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_183);  sub_54 = unsqueeze_183 = None
    mul_267: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_266, relu_22, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = primals_95 = None
    getitem_85: "f32[8, 512, 8, 8]" = convolution_backward_4[0]
    getitem_86: "f32[2048, 512, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_44: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_45: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_4: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_85);  le_4 = scalar_tensor_4 = getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_185: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_55: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186)
    mul_268: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_55);  sub_55 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 2, 3]);  mul_268 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_187: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_188: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_270: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_190: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_191: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_273: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_55);  primals_55 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    sub_56: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186);  avg_pool2d = unsqueeze_186 = None
    mul_274: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_192);  sub_56 = unsqueeze_192 = None
    sub_57: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_274);  where_4 = mul_274 = None
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_189);  sub_57 = unsqueeze_189 = None
    mul_275: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_195);  sub_58 = unsqueeze_195 = None
    mul_276: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d_backward: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(mul_275, view_47, [2, 2], [2, 2], [0, 0], False, True, None);  mul_275 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_99: "f32[32, 128, 256]" = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
    permute_47: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    view_100: "f32[32, 256, 128]" = torch.ops.aten.view.default(permute_47, [32, 256, 128]);  permute_47 = None
    permute_48: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_10: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(permute_48, view_100);  permute_48 = None
    permute_49: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_11: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_100, permute_49);  view_100 = permute_49 = None
    view_101: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_10, [32, 256, 128]);  bmm_10 = None
    view_102: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_11, [32, 256, 256]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_46: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_277: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_102, alias_46);  view_102 = None
    sum_20: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
    mul_278: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_46, sum_20);  alias_46 = sum_20 = None
    sub_59: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_103: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_59, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_50: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_103, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_21: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_50, [2], True);  permute_50 = None
    view_104: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_21, [512, 16, 16]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_6: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_6: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_6, view_104, 2, 15, 9223372036854775807);  full_6 = view_104 = None
    full_7: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_7: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_6, 1, 0, 16);  full_7 = slice_scatter_6 = None
    full_8: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_8: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_7, 0, 0, 9223372036854775807);  full_8 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_105: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_16: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_105, [0, -15]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_106: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_16, [512, 16, 32]);  constant_pad_nd_16 = None
    constant_pad_nd_17: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, -1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_107: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_17, [32, 16, 16, 31]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_108: "f32[8192, 31]" = torch.ops.aten.view.default(view_107, [8192, 31]);  view_107 = None
    permute_51: "f32[31, 8192]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_12: "f32[31, 128]" = torch.ops.aten.mm.default(permute_51, view_37);  permute_51 = view_37 = None
    permute_52: "f32[128, 31]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    permute_53: "f32[31, 128]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_13: "f32[8192, 128]" = torch.ops.aten.mm.default(view_108, permute_53);  view_108 = permute_53 = None
    view_109: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_13, [32, 16, 16, 128]);  mm_13 = None
    permute_54: "f32[31, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_55: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_56: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_22: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_56, [2], True);  permute_56 = None
    view_110: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_22, [512, 16, 16]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_9: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_9: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_9, view_110, 2, 15, 9223372036854775807);  full_9 = view_110 = None
    full_10: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_10: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_9, 1, 0, 16);  full_10 = slice_scatter_9 = None
    full_11: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_11: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_11, slice_scatter_10, 0, 0, 9223372036854775807);  full_11 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_111: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_18: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_111, [0, -15]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_112: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_18, [512, 16, 32]);  constant_pad_nd_18 = None
    constant_pad_nd_19: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, -1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_113: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_19, [32, 16, 16, 31]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_114: "f32[8192, 31]" = torch.ops.aten.view.default(view_113, [8192, 31]);  view_113 = None
    permute_57: "f32[31, 8192]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_14: "f32[31, 128]" = torch.ops.aten.mm.default(permute_57, view_31);  permute_57 = view_31 = None
    permute_58: "f32[128, 31]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    permute_59: "f32[31, 128]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_15: "f32[8192, 128]" = torch.ops.aten.mm.default(view_114, permute_59);  view_114 = permute_59 = None
    view_115: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_15, [32, 16, 16, 128]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_172: "f32[32, 16, 16, 128]" = torch.ops.aten.add.Tensor(permute_55, view_115);  permute_55 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_60: "f32[31, 128]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_25: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(add_172, memory_format = torch.contiguous_format);  add_172 = None
    view_116: "f32[32, 256, 128]" = torch.ops.aten.view.default(clone_25, [32, 256, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_279: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_59, 0.08838834764831845);  sub_59 = None
    view_117: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_279, [32, 256, 256]);  mul_279 = None
    permute_61: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_12: "f32[32, 128, 256]" = torch.ops.aten.bmm.default(permute_61, view_117);  permute_61 = None
    permute_62: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_13: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(view_117, permute_62);  view_117 = permute_62 = None
    view_118: "f32[32, 128, 256]" = torch.ops.aten.view.default(bmm_12, [32, 128, 256]);  bmm_12 = None
    view_119: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_13, [32, 256, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_173: "f32[32, 256, 128]" = torch.ops.aten.add.Tensor(view_116, view_119);  view_116 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_63: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    clone_26: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_120: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_26, [8, 512, 16, 16]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_121: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(view_118, [8, 512, 16, 16]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_64: "f32[32, 128, 256]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    clone_27: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_122: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_27, [8, 512, 16, 16]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_1: "f32[8, 1536, 16, 16]" = torch.ops.aten.cat.default([view_122, view_121, view_120], 1);  view_122 = view_121 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_1, relu_21, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = primals_94 = None
    getitem_88: "f32[8, 512, 16, 16]" = convolution_backward_5[0]
    getitem_89: "f32[1536, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_49: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_5: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_88);  le_5 = scalar_tensor_5 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_60: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_198)
    mul_280: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_5, sub_60);  sub_60 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 2, 3]);  mul_280 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
    unsqueeze_199: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_200: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    unsqueeze_202: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_284, 0);  mul_284 = None
    unsqueeze_203: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    sub_61: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_198);  convolution_24 = unsqueeze_198 = None
    mul_286: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_204);  sub_61 = unsqueeze_204 = None
    sub_62: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_5, mul_286);  where_5 = mul_286 = None
    sub_63: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_201);  sub_62 = unsqueeze_201 = None
    mul_287: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_207);  sub_63 = unsqueeze_207 = None
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_287, relu_20, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_287 = primals_93 = None
    getitem_91: "f32[8, 1024, 16, 16]" = convolution_backward_6[0]
    getitem_92: "f32[512, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_174: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_82, getitem_91);  getitem_82 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    alias_51: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_52: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_6: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_174);  le_6 = scalar_tensor_6 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_209: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_64: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_210)
    mul_289: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, sub_64);  sub_64 = None
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_290: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
    unsqueeze_211: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_212: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_291: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_293: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_214: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_293, 0);  mul_293 = None
    unsqueeze_215: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_294: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_217: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_218: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_65: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_210);  convolution_23 = unsqueeze_210 = None
    mul_295: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_216);  sub_65 = unsqueeze_216 = None
    sub_66: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_6, mul_295);  mul_295 = None
    sub_67: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_213);  sub_66 = unsqueeze_213 = None
    mul_296: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_219);  sub_67 = unsqueeze_219 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_296, relu_19, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_296 = primals_92 = None
    getitem_94: "f32[8, 256, 16, 16]" = convolution_backward_7[0]
    getitem_95: "f32[1024, 256, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_55: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_7: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_94);  le_7 = scalar_tensor_7 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_68: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_222)
    mul_298: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_7, sub_68);  sub_68 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3]);  mul_298 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_224: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    mul_301: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_302: "f32[256]" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_226: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_227: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_303: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_69: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_222);  view_23 = unsqueeze_222 = None
    mul_304: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_228);  sub_69 = unsqueeze_228 = None
    sub_70: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_7, mul_304);  where_7 = mul_304 = None
    sub_71: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_225);  sub_70 = unsqueeze_225 = None
    mul_305: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_231);  sub_71 = unsqueeze_231 = None
    mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_123: "f32[32, 64, 256]" = torch.ops.aten.view.default(mul_305, [32, 64, 256]);  mul_305 = None
    permute_65: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    view_124: "f32[32, 256, 64]" = torch.ops.aten.view.default(permute_65, [32, 256, 64]);  permute_65 = None
    permute_66: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    bmm_14: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(permute_66, view_124);  permute_66 = None
    permute_67: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    bmm_15: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_124, permute_67);  view_124 = permute_67 = None
    view_125: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_14, [32, 256, 64]);  bmm_14 = None
    view_126: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_15, [32, 256, 256]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_56: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_307: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_126, alias_56);  view_126 = None
    sum_29: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
    mul_308: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_56, sum_29);  alias_56 = sum_29 = None
    sub_72: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_127: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_72, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_68: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_30: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_68, [2], True);  permute_68 = None
    view_128: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_30, [512, 16, 16]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_12: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_12: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_12, view_128, 2, 15, 9223372036854775807);  full_12 = view_128 = None
    full_13: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_13: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_13, slice_scatter_12, 1, 0, 16);  full_13 = slice_scatter_12 = None
    full_14: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_14: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_13, 0, 0, 9223372036854775807);  full_14 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_129: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_20: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_129, [0, -15]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_130: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, -1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_131: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_132: "f32[8192, 31]" = torch.ops.aten.view.default(view_131, [8192, 31]);  view_131 = None
    permute_69: "f32[31, 8192]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_16: "f32[31, 64]" = torch.ops.aten.mm.default(permute_69, view_13);  permute_69 = view_13 = None
    permute_70: "f32[64, 31]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    permute_71: "f32[31, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_17: "f32[8192, 64]" = torch.ops.aten.mm.default(view_132, permute_71);  view_132 = permute_71 = None
    view_133: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_17, [32, 16, 16, 64]);  mm_17 = None
    permute_72: "f32[31, 64]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_73: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_74: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2, 4]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_31: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_74, [2], True);  permute_74 = None
    view_134: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_31, [512, 16, 16]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_15: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_15: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_15, view_134, 2, 15, 9223372036854775807);  full_15 = view_134 = None
    full_16: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_16: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_16, slice_scatter_15, 1, 0, 16);  full_16 = slice_scatter_15 = None
    full_17: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_17: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_17, slice_scatter_16, 0, 0, 9223372036854775807);  full_17 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_135: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_22: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, -15]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_136: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, -1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_137: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_138: "f32[8192, 31]" = torch.ops.aten.view.default(view_137, [8192, 31]);  view_137 = None
    permute_75: "f32[31, 8192]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_18: "f32[31, 64]" = torch.ops.aten.mm.default(permute_75, view_7);  permute_75 = view_7 = None
    permute_76: "f32[64, 31]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    permute_77: "f32[31, 64]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_19: "f32[8192, 64]" = torch.ops.aten.mm.default(view_138, permute_77);  view_138 = permute_77 = None
    view_139: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_19, [32, 16, 16, 64]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_175: "f32[32, 16, 16, 64]" = torch.ops.aten.add.Tensor(permute_73, view_139);  permute_73 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_78: "f32[31, 64]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_28: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(add_175, memory_format = torch.contiguous_format);  add_175 = None
    view_140: "f32[32, 256, 64]" = torch.ops.aten.view.default(clone_28, [32, 256, 64]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_309: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_72, 0.125);  sub_72 = None
    view_141: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_309, [32, 256, 256]);  mul_309 = None
    permute_79: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    bmm_16: "f32[32, 64, 256]" = torch.ops.aten.bmm.default(permute_79, view_141);  permute_79 = None
    permute_80: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    bmm_17: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(view_141, permute_80);  view_141 = permute_80 = None
    view_142: "f32[32, 64, 256]" = torch.ops.aten.view.default(bmm_16, [32, 64, 256]);  bmm_16 = None
    view_143: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_17, [32, 256, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_176: "f32[32, 256, 64]" = torch.ops.aten.add.Tensor(view_140, view_143);  view_140 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_81: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    clone_29: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    view_144: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_29, [8, 256, 16, 16]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_145: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(view_142, [8, 256, 16, 16]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_82: "f32[32, 64, 256]" = torch.ops.aten.permute.default(add_176, [0, 2, 1]);  add_176 = None
    clone_30: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_146: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_30, [8, 256, 16, 16]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_2: "f32[8, 768, 16, 16]" = torch.ops.aten.cat.default([view_146, view_145, view_144], 1);  view_146 = view_145 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_2, relu_18, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = primals_91 = None
    getitem_97: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_98: "f32[768, 256, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_58: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_59: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_8: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_97);  le_8 = scalar_tensor_8 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_73: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_234)
    mul_310: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_8, sub_73);  sub_73 = None
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 2, 3]);  mul_310 = None
    mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_235: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_311, 0);  mul_311 = None
    unsqueeze_236: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_239: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_74: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_234);  convolution_21 = unsqueeze_234 = None
    mul_316: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_240);  sub_74 = unsqueeze_240 = None
    sub_75: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_8, mul_316);  where_8 = mul_316 = None
    sub_76: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_237);  sub_75 = unsqueeze_237 = None
    mul_317: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_243);  sub_76 = unsqueeze_243 = None
    mul_318: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_317, relu_17, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_317 = primals_90 = None
    getitem_100: "f32[8, 1024, 16, 16]" = convolution_backward_9[0]
    getitem_101: "f32[256, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_177: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_6, getitem_100);  where_6 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_61: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_62: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    le_9: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_62, 0);  alias_62 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_177);  le_9 = scalar_tensor_9 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_245: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_77: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_246)
    mul_319: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_77);  sub_77 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 2, 3]);  mul_319 = None
    mul_320: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_247: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_248: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_321: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_322: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_323: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    unsqueeze_250: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_323, 0);  mul_323 = None
    unsqueeze_251: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_324: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_253: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_254: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_78: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_246);  convolution_20 = unsqueeze_246 = None
    mul_325: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_252);  sub_78 = unsqueeze_252 = None
    sub_79: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_325);  mul_325 = None
    sub_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_249);  sub_79 = unsqueeze_249 = None
    mul_326: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_255);  sub_80 = unsqueeze_255 = None
    mul_327: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_326, relu_14, primals_89, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_326 = primals_89 = None
    getitem_103: "f32[8, 512, 32, 32]" = convolution_backward_10[0]
    getitem_104: "f32[1024, 512, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_257: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_36: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_81: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_258)
    mul_328: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_81);  sub_81 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 2, 3]);  mul_328 = None
    mul_329: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    unsqueeze_259: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
    unsqueeze_260: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_330: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_331: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_332: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    unsqueeze_262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_333: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_82: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_258);  convolution_19 = unsqueeze_258 = None
    mul_334: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_264);  sub_82 = unsqueeze_264 = None
    sub_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_334);  where_9 = mul_334 = None
    sub_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_261);  sub_83 = unsqueeze_261 = None
    mul_335: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_267);  sub_84 = unsqueeze_267 = None
    mul_336: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_335, relu_16, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_335 = primals_88 = None
    getitem_106: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_107: "f32[1024, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_64: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_65: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    le_10: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_106);  le_10 = scalar_tensor_10 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_269: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_85: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_270)
    mul_337: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_85);  sub_85 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    unsqueeze_271: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_272: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    mul_340: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_341: "f32[256]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_274: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_275: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_342: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_277: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_278: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_86: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_270);  convolution_18 = unsqueeze_270 = None
    mul_343: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_276);  sub_86 = unsqueeze_276 = None
    sub_87: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_343);  where_10 = mul_343 = None
    sub_88: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_273);  sub_87 = unsqueeze_273 = None
    mul_344: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_279);  sub_88 = unsqueeze_279 = None
    mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_55);  sum_39 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_344, relu_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_344 = primals_87 = None
    getitem_109: "f32[8, 256, 32, 32]" = convolution_backward_12[0]
    getitem_110: "f32[256, 256, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_67: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_68: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    le_11: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_68, 0);  alias_68 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_109);  le_11 = scalar_tensor_11 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_281: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_89: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_282)
    mul_346: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_11, sub_89);  sub_89 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 2, 3]);  mul_346 = None
    mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.0001220703125)
    unsqueeze_283: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_284: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
    mul_349: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_350: "f32[256]" = torch.ops.aten.mul.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    unsqueeze_286: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_287: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_351: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_289: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_290: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_90: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_282);  convolution_17 = unsqueeze_282 = None
    mul_352: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_288);  sub_90 = unsqueeze_288 = None
    sub_91: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_11, mul_352);  where_11 = mul_352 = None
    sub_92: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_285);  sub_91 = unsqueeze_285 = None
    mul_353: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_291);  sub_92 = unsqueeze_291 = None
    mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_52);  sum_41 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_353, relu_14, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_353 = primals_86 = None
    getitem_112: "f32[8, 512, 32, 32]" = convolution_backward_13[0]
    getitem_113: "f32[256, 512, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_178: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_103, getitem_112);  getitem_103 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_70: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_71: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_12: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_178);  le_12 = scalar_tensor_12 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_293: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_93: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_294)
    mul_355: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_12, sub_93);  sub_93 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 2, 3]);  mul_355 = None
    mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
    unsqueeze_295: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_296: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_357: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
    mul_358: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_359: "f32[512]" = torch.ops.aten.mul.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    unsqueeze_298: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_299: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_301: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
    unsqueeze_302: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_94: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_294);  convolution_16 = unsqueeze_294 = None
    mul_361: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_300);  sub_94 = unsqueeze_300 = None
    sub_95: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_12, mul_361);  mul_361 = None
    sub_96: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_297);  sub_95 = unsqueeze_297 = None
    mul_362: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_303);  sub_96 = unsqueeze_303 = None
    mul_363: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_49);  sum_43 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_362, relu_13, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_362 = primals_85 = None
    getitem_115: "f32[8, 128, 32, 32]" = convolution_backward_14[0]
    getitem_116: "f32[512, 128, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_73: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_74: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    le_13: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_74, 0);  alias_74 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_115);  le_13 = scalar_tensor_13 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_305: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_97: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_306)
    mul_364: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_13, sub_97);  sub_97 = None
    sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[128]" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
    unsqueeze_307: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_308: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_366: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, 0.0001220703125)
    mul_367: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_310: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_311: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_369: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_313: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_314: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_98: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_306);  convolution_15 = unsqueeze_306 = None
    mul_370: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_312);  sub_98 = unsqueeze_312 = None
    sub_99: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_13, mul_370);  where_13 = mul_370 = None
    sub_100: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_309);  sub_99 = unsqueeze_309 = None
    mul_371: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_315);  sub_100 = unsqueeze_315 = None
    mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_46);  sum_45 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_371, relu_12, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = primals_84 = None
    getitem_118: "f32[8, 128, 32, 32]" = convolution_backward_15[0]
    getitem_119: "f32[128, 128, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_76: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_77: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    le_14: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_118);  le_14 = scalar_tensor_14 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_101: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_318)
    mul_373: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_14, sub_101);  sub_101 = None
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_374: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
    unsqueeze_319: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_320: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
    mul_376: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_377: "f32[128]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_322: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_323: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_378: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_102: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_318);  convolution_14 = unsqueeze_318 = None
    mul_379: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_324);  sub_102 = unsqueeze_324 = None
    sub_103: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_14, mul_379);  where_14 = mul_379 = None
    sub_104: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_321);  sub_103 = unsqueeze_321 = None
    mul_380: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_327);  sub_104 = unsqueeze_327 = None
    mul_381: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_43);  sum_47 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_380, relu_11, primals_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = primals_83 = None
    getitem_121: "f32[8, 512, 32, 32]" = convolution_backward_16[0]
    getitem_122: "f32[128, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_179: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(where_12, getitem_121);  where_12 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_79: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_80: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    le_15: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_80, 0);  alias_80 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_179);  le_15 = scalar_tensor_15 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_329: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_105: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_330)
    mul_382: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_105);  sub_105 = None
    sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
    unsqueeze_331: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_332: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_384: "f32[512]" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
    mul_385: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_386: "f32[512]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_334: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_335: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_387: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_337: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_338: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_106: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_330);  convolution_13 = unsqueeze_330 = None
    mul_388: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_336);  sub_106 = unsqueeze_336 = None
    sub_107: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_388);  mul_388 = None
    sub_108: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_333);  sub_107 = unsqueeze_333 = None
    mul_389: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_339);  sub_108 = unsqueeze_339 = None
    mul_390: "f32[512]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_40);  sum_49 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_389, relu_8, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_389 = primals_82 = None
    getitem_124: "f32[8, 256, 64, 64]" = convolution_backward_17[0]
    getitem_125: "f32[512, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_341: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_109: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_342)
    mul_391: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_109);  sub_109 = None
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_392: "f32[512]" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
    unsqueeze_343: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_344: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_393: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    mul_394: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_395: "f32[512]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_346: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_347: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_396: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_349: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_350: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_110: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_342);  convolution_12 = unsqueeze_342 = None
    mul_397: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_348);  sub_110 = unsqueeze_348 = None
    sub_111: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_397);  where_15 = mul_397 = None
    sub_112: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_345);  sub_111 = unsqueeze_345 = None
    mul_398: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_351);  sub_112 = unsqueeze_351 = None
    mul_399: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_37);  sum_51 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_398, relu_10, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = primals_81 = None
    getitem_127: "f32[8, 128, 32, 32]" = convolution_backward_18[0]
    getitem_128: "f32[512, 128, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_82: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_83: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le_16: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_127);  le_16 = scalar_tensor_16 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_353: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_52: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_113: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_354)
    mul_400: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_16, sub_113);  sub_113 = None
    sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[128]" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
    unsqueeze_355: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_356: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_402: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    mul_403: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_404: "f32[128]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_358: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_359: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_405: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_361: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_362: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_114: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_354);  convolution_11 = unsqueeze_354 = None
    mul_406: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_360);  sub_114 = unsqueeze_360 = None
    sub_115: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_16, mul_406);  where_16 = mul_406 = None
    sub_116: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_357);  sub_115 = unsqueeze_357 = None
    mul_407: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_363);  sub_116 = unsqueeze_363 = None
    mul_408: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_34);  sum_53 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_407, relu_9, primals_80, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_80 = None
    getitem_130: "f32[8, 128, 64, 64]" = convolution_backward_19[0]
    getitem_131: "f32[128, 128, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_85: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_86: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_17: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_86, 0);  alias_86 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_130);  le_17 = scalar_tensor_17 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_365: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_117: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_366)
    mul_409: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_17, sub_117);  sub_117 = None
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_410: "f32[128]" = torch.ops.aten.mul.Tensor(sum_54, 3.0517578125e-05)
    unsqueeze_367: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_368: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_411: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 3.0517578125e-05)
    mul_412: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_413: "f32[128]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    unsqueeze_370: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_371: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_414: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_373: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_374: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_118: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_366);  convolution_10 = unsqueeze_366 = None
    mul_415: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_372);  sub_118 = unsqueeze_372 = None
    sub_119: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_17, mul_415);  where_17 = mul_415 = None
    sub_120: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_369);  sub_119 = unsqueeze_369 = None
    mul_416: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_375);  sub_120 = unsqueeze_375 = None
    mul_417: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_31);  sum_55 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_416, relu_8, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_416 = primals_79 = None
    getitem_133: "f32[8, 256, 64, 64]" = convolution_backward_20[0]
    getitem_134: "f32[128, 256, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_180: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_124, getitem_133);  getitem_124 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_88: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_89: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_18: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_180);  le_18 = scalar_tensor_18 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_377: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_121: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_378)
    mul_418: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_18, sub_121);  sub_121 = None
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 3.0517578125e-05)
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_422: "f32[256]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_423: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_122: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_378);  convolution_9 = unsqueeze_378 = None
    mul_424: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_384);  sub_122 = unsqueeze_384 = None
    sub_123: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_18, mul_424);  mul_424 = None
    sub_124: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_381);  sub_123 = unsqueeze_381 = None
    mul_425: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_387);  sub_124 = unsqueeze_387 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_28);  sum_57 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_425, relu_7, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = primals_78 = None
    getitem_136: "f32[8, 64, 64, 64]" = convolution_backward_21[0]
    getitem_137: "f32[256, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_91: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_92: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_19: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_136);  le_19 = scalar_tensor_19 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_389: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_58: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_125: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_390)
    mul_427: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_19, sub_125);  sub_125 = None
    sum_59: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[64]" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
    unsqueeze_391: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_392: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_429: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
    mul_430: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_431: "f32[64]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_394: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_395: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_432: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_397: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_398: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_126: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_390);  convolution_8 = unsqueeze_390 = None
    mul_433: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_396);  sub_126 = unsqueeze_396 = None
    sub_127: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_19, mul_433);  where_19 = mul_433 = None
    sub_128: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_393);  sub_127 = unsqueeze_393 = None
    mul_434: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_399);  sub_128 = unsqueeze_399 = None
    mul_435: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_25);  sum_59 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_434, relu_6, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_77 = None
    getitem_139: "f32[8, 64, 64, 64]" = convolution_backward_22[0]
    getitem_140: "f32[64, 64, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_94: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_95: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_20: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_139);  le_20 = scalar_tensor_20 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_401: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_60: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_129: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_402)
    mul_436: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_20, sub_129);  sub_129 = None
    sum_61: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[64]" = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
    unsqueeze_403: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_404: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_438: "f32[64]" = torch.ops.aten.mul.Tensor(sum_61, 3.0517578125e-05)
    mul_439: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_440: "f32[64]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_406: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_407: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_441: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_409: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_410: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_130: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_402);  convolution_7 = unsqueeze_402 = None
    mul_442: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_408);  sub_130 = unsqueeze_408 = None
    sub_131: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_20, mul_442);  where_20 = mul_442 = None
    sub_132: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_405);  sub_131 = unsqueeze_405 = None
    mul_443: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_411);  sub_132 = unsqueeze_411 = None
    mul_444: "f32[64]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_22);  sum_61 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_443, relu_5, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = primals_76 = None
    getitem_142: "f32[8, 256, 64, 64]" = convolution_backward_23[0]
    getitem_143: "f32[64, 256, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_181: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(where_18, getitem_142);  where_18 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_97: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_98: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_21: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_181);  le_21 = scalar_tensor_21 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_413: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_133: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_414)
    mul_445: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_21, sub_133);  sub_133 = None
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[256]" = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
    unsqueeze_415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
    mul_448: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_449: "f32[256]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_418: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_419: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_450: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_421: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_422: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_134: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_414);  convolution_6 = unsqueeze_414 = None
    mul_451: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_420);  sub_134 = unsqueeze_420 = None
    sub_135: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_21, mul_451);  mul_451 = None
    sub_136: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_417);  sub_135 = unsqueeze_417 = None
    mul_452: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_423);  sub_136 = unsqueeze_423 = None
    mul_453: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_19);  sum_63 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_452, getitem_6, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = primals_75 = None
    getitem_145: "f32[8, 64, 64, 64]" = convolution_backward_24[0]
    getitem_146: "f32[256, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_425: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_64: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_137: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_426)
    mul_454: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_21, sub_137);  sub_137 = None
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_455: "f32[256]" = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_456: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
    mul_457: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_458: "f32[256]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_430: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_431: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_459: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_433: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_434: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_138: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_426);  convolution_5 = unsqueeze_426 = None
    mul_460: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_432);  sub_138 = unsqueeze_432 = None
    sub_139: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_21, mul_460);  where_21 = mul_460 = None
    sub_140: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_429);  sub_139 = unsqueeze_429 = None
    mul_461: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_435);  sub_140 = unsqueeze_435 = None
    mul_462: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_16);  sum_65 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_461, relu_4, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_74 = None
    getitem_148: "f32[8, 64, 64, 64]" = convolution_backward_25[0]
    getitem_149: "f32[256, 64, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_100: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_101: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    le_22: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_101, 0);  alias_101 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_148);  le_22 = scalar_tensor_22 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_437: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_66: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_141: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_438)
    mul_463: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_22, sub_141);  sub_141 = None
    sum_67: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[64]" = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
    unsqueeze_439: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_440: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_465: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
    mul_466: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_467: "f32[64]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_442: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_443: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_468: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_445: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_446: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_142: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_438);  convolution_4 = unsqueeze_438 = None
    mul_469: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_444);  sub_142 = unsqueeze_444 = None
    sub_143: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_22, mul_469);  where_22 = mul_469 = None
    sub_144: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_441);  sub_143 = unsqueeze_441 = None
    mul_470: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_447);  sub_144 = unsqueeze_447 = None
    mul_471: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_13);  sum_67 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_470, relu_3, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_470 = primals_73 = None
    getitem_151: "f32[8, 64, 64, 64]" = convolution_backward_26[0]
    getitem_152: "f32[64, 64, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_103: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_104: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_23: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_151);  le_23 = scalar_tensor_23 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_449: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_68: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_145: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_450)
    mul_472: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_23, sub_145);  sub_145 = None
    sum_69: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_473: "f32[64]" = torch.ops.aten.mul.Tensor(sum_68, 3.0517578125e-05)
    unsqueeze_451: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_452: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_474: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
    mul_475: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_476: "f32[64]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_454: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_455: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_477: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_457: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_458: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_146: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_450);  convolution_3 = unsqueeze_450 = None
    mul_478: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_456);  sub_146 = unsqueeze_456 = None
    sub_147: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_23, mul_478);  where_23 = mul_478 = None
    sub_148: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_453);  sub_147 = unsqueeze_453 = None
    mul_479: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_459);  sub_148 = unsqueeze_459 = None
    mul_480: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_10);  sum_69 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_479, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = getitem_6 = primals_72 = None
    getitem_154: "f32[8, 64, 64, 64]" = convolution_backward_27[0]
    getitem_155: "f32[64, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_182: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_145, getitem_154);  getitem_145 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_182, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7);  add_182 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_106: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_107: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_24: "b8[8, 64, 128, 128]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, max_pool2d_with_indices_backward);  le_24 = scalar_tensor_24 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_461: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_149: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_462)
    mul_481: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_24, sub_149);  sub_149 = None
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_482: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 7.62939453125e-06)
    unsqueeze_463: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_464: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_483: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 7.62939453125e-06)
    mul_484: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_485: "f32[64]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_466: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_467: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_486: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_469: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_470: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_150: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_462);  convolution_2 = unsqueeze_462 = None
    mul_487: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_468);  sub_150 = unsqueeze_468 = None
    sub_151: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_24, mul_487);  where_24 = mul_487 = None
    sub_152: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_465);  sub_151 = unsqueeze_465 = None
    mul_488: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_471);  sub_152 = unsqueeze_471 = None
    mul_489: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_7);  sum_71 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_488, relu_1, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_71 = None
    getitem_157: "f32[8, 32, 128, 128]" = convolution_backward_28[0]
    getitem_158: "f32[64, 32, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_109: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_110: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_25: "b8[8, 32, 128, 128]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_157);  le_25 = scalar_tensor_25 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_473: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_72: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_153: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_474)
    mul_490: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(where_25, sub_153);  sub_153 = None
    sum_73: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_491: "f32[32]" = torch.ops.aten.mul.Tensor(sum_72, 7.62939453125e-06)
    unsqueeze_475: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_476: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_492: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, 7.62939453125e-06)
    mul_493: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_494: "f32[32]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_478: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_479: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_495: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_481: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_482: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_154: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_474);  convolution_1 = unsqueeze_474 = None
    mul_496: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_480);  sub_154 = unsqueeze_480 = None
    sub_155: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(where_25, mul_496);  where_25 = mul_496 = None
    sub_156: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_477);  sub_155 = unsqueeze_477 = None
    mul_497: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_483);  sub_156 = unsqueeze_483 = None
    mul_498: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_4);  sum_73 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_497, relu, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_497 = primals_70 = None
    getitem_160: "f32[8, 24, 128, 128]" = convolution_backward_29[0]
    getitem_161: "f32[32, 24, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_112: "f32[8, 24, 128, 128]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_113: "f32[8, 24, 128, 128]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_26: "b8[8, 24, 128, 128]" = torch.ops.aten.le.Scalar(alias_113, 0);  alias_113 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[8, 24, 128, 128]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_160);  le_26 = scalar_tensor_26 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_485: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_74: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_157: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_486)
    mul_499: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(where_26, sub_157);  sub_157 = None
    sum_75: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_500: "f32[24]" = torch.ops.aten.mul.Tensor(sum_74, 7.62939453125e-06)
    unsqueeze_487: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_488: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_501: "f32[24]" = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
    mul_502: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_503: "f32[24]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_490: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_491: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_504: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_493: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_494: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_158: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_486);  convolution = unsqueeze_486 = None
    mul_505: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_492);  sub_158 = unsqueeze_492 = None
    sub_159: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(where_26, mul_505);  where_26 = mul_505 = None
    sub_160: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_489);  sub_159 = unsqueeze_489 = None
    mul_506: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_495);  sub_160 = unsqueeze_495 = None
    mul_507: "f32[24]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_1);  sum_75 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_506, primals_195, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_506 = primals_195 = primals_69 = None
    getitem_164: "f32[24, 3, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_102, add);  primals_102 = add = None
    copy__1: "f32[24]" = torch.ops.aten.copy_.default(primals_103, add_2);  primals_103 = add_2 = None
    copy__2: "f32[24]" = torch.ops.aten.copy_.default(primals_104, add_3);  primals_104 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_5);  primals_105 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_106, add_7);  primals_106 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_107, add_8);  primals_107 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_108, add_10);  primals_108 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_109, add_12);  primals_109 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_110, add_13);  primals_110 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_15);  primals_111 = add_15 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_112, add_17);  primals_112 = add_17 = None
    copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_113, add_18);  primals_113 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_114, add_20);  primals_114 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_115, add_22);  primals_115 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_116, add_23);  primals_116 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_117, add_25);  primals_117 = add_25 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_118, add_27);  primals_118 = add_27 = None
    copy__17: "f32[256]" = torch.ops.aten.copy_.default(primals_119, add_28);  primals_119 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_120, add_30);  primals_120 = add_30 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_121, add_32);  primals_121 = add_32 = None
    copy__20: "f32[256]" = torch.ops.aten.copy_.default(primals_122, add_33);  primals_122 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_36);  primals_123 = add_36 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_124, add_38);  primals_124 = add_38 = None
    copy__23: "f32[64]" = torch.ops.aten.copy_.default(primals_125, add_39);  primals_125 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_126, add_41);  primals_126 = add_41 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_127, add_43);  primals_127 = add_43 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_128, add_44);  primals_128 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_46);  primals_129 = add_46 = None
    copy__28: "f32[256]" = torch.ops.aten.copy_.default(primals_130, add_48);  primals_130 = add_48 = None
    copy__29: "f32[256]" = torch.ops.aten.copy_.default(primals_131, add_49);  primals_131 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_132, add_52);  primals_132 = add_52 = None
    copy__31: "f32[128]" = torch.ops.aten.copy_.default(primals_133, add_54);  primals_133 = add_54 = None
    copy__32: "f32[128]" = torch.ops.aten.copy_.default(primals_134, add_55);  primals_134 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_57);  primals_135 = add_57 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_136, add_59);  primals_136 = add_59 = None
    copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_137, add_60);  primals_137 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_138, add_62);  primals_138 = add_62 = None
    copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_139, add_64);  primals_139 = add_64 = None
    copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_140, add_65);  primals_140 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_67);  primals_141 = add_67 = None
    copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_142, add_69);  primals_142 = add_69 = None
    copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_143, add_70);  primals_143 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_144, add_73);  primals_144 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_145, add_75);  primals_145 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_146, add_76);  primals_146 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_78);  primals_147 = add_78 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_148, add_80);  primals_148 = add_80 = None
    copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_149, add_81);  primals_149 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_150, add_83);  primals_150 = add_83 = None
    copy__49: "f32[512]" = torch.ops.aten.copy_.default(primals_151, add_85);  primals_151 = add_85 = None
    copy__50: "f32[512]" = torch.ops.aten.copy_.default(primals_152, add_86);  primals_152 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_153, add_89);  primals_153 = add_89 = None
    copy__52: "f32[256]" = torch.ops.aten.copy_.default(primals_154, add_91);  primals_154 = add_91 = None
    copy__53: "f32[256]" = torch.ops.aten.copy_.default(primals_155, add_92);  primals_155 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_156, add_94);  primals_156 = add_94 = None
    copy__55: "f32[256]" = torch.ops.aten.copy_.default(primals_157, add_96);  primals_157 = add_96 = None
    copy__56: "f32[256]" = torch.ops.aten.copy_.default(primals_158, add_97);  primals_158 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_159, add_99);  primals_159 = add_99 = None
    copy__58: "f32[1024]" = torch.ops.aten.copy_.default(primals_160, add_101);  primals_160 = add_101 = None
    copy__59: "f32[1024]" = torch.ops.aten.copy_.default(primals_161, add_102);  primals_161 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_162, add_104);  primals_162 = add_104 = None
    copy__61: "f32[1024]" = torch.ops.aten.copy_.default(primals_163, add_106);  primals_163 = add_106 = None
    copy__62: "f32[1024]" = torch.ops.aten.copy_.default(primals_164, add_107);  primals_164 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_110);  primals_165 = add_110 = None
    copy__64: "f32[256]" = torch.ops.aten.copy_.default(primals_166, add_112);  primals_166 = add_112 = None
    copy__65: "f32[256]" = torch.ops.aten.copy_.default(primals_167, add_113);  primals_167 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_168, add_117);  primals_168 = add_117 = None
    copy__67: "f32[256]" = torch.ops.aten.copy_.default(primals_169, add_119);  primals_169 = add_119 = None
    copy__68: "f32[256]" = torch.ops.aten.copy_.default(primals_170, add_120);  primals_170 = add_120 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_171, add_122);  primals_171 = add_122 = None
    copy__70: "f32[1024]" = torch.ops.aten.copy_.default(primals_172, add_124);  primals_172 = add_124 = None
    copy__71: "f32[1024]" = torch.ops.aten.copy_.default(primals_173, add_125);  primals_173 = add_125 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_174, add_128);  primals_174 = add_128 = None
    copy__73: "f32[512]" = torch.ops.aten.copy_.default(primals_175, add_130);  primals_175 = add_130 = None
    copy__74: "f32[512]" = torch.ops.aten.copy_.default(primals_176, add_131);  primals_176 = add_131 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_135);  primals_177 = add_135 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_178, add_137);  primals_178 = add_137 = None
    copy__77: "f32[512]" = torch.ops.aten.copy_.default(primals_179, add_138);  primals_179 = add_138 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_180, add_140);  primals_180 = add_140 = None
    copy__79: "f32[2048]" = torch.ops.aten.copy_.default(primals_181, add_142);  primals_181 = add_142 = None
    copy__80: "f32[2048]" = torch.ops.aten.copy_.default(primals_182, add_143);  primals_182 = add_143 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_145);  primals_183 = add_145 = None
    copy__82: "f32[2048]" = torch.ops.aten.copy_.default(primals_184, add_147);  primals_184 = add_147 = None
    copy__83: "f32[2048]" = torch.ops.aten.copy_.default(primals_185, add_148);  primals_185 = add_148 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_186, add_151);  primals_186 = add_151 = None
    copy__85: "f32[512]" = torch.ops.aten.copy_.default(primals_187, add_153);  primals_187 = add_153 = None
    copy__86: "f32[512]" = torch.ops.aten.copy_.default(primals_188, add_154);  primals_188 = add_154 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_158);  primals_189 = add_158 = None
    copy__88: "f32[512]" = torch.ops.aten.copy_.default(primals_190, add_160);  primals_190 = add_160 = None
    copy__89: "f32[512]" = torch.ops.aten.copy_.default(primals_191, add_161);  primals_191 = add_161 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_192, add_163);  primals_192 = add_163 = None
    copy__91: "f32[2048]" = torch.ops.aten.copy_.default(primals_193, add_165);  primals_193 = add_165 = None
    copy__92: "f32[2048]" = torch.ops.aten.copy_.default(primals_194, add_166);  primals_194 = add_166 = None
    return pytree.tree_unflatten([addmm, mul_507, sum_74, mul_498, sum_72, mul_489, sum_70, mul_480, sum_68, mul_471, sum_66, mul_462, sum_64, mul_453, sum_62, mul_444, sum_60, mul_435, sum_58, mul_426, sum_56, mul_417, sum_54, mul_408, sum_52, mul_399, sum_50, mul_390, sum_48, mul_381, sum_46, mul_372, sum_44, mul_363, sum_42, mul_354, sum_40, mul_345, sum_38, mul_336, sum_36, mul_327, sum_34, mul_318, sum_32, permute_78, permute_72, mul_306, sum_27, mul_297, sum_25, mul_288, sum_23, permute_60, permute_54, mul_276, sum_18, mul_267, sum_16, mul_258, sum_14, mul_249, sum_12, permute_42, permute_36, mul_237, sum_7, mul_228, sum_5, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, getitem_77, getitem_74, permute_28, view_73, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    