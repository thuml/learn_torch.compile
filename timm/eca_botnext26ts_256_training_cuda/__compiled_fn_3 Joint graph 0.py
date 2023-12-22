from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[24]"; primals_2: "f32[24]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64]"; primals_11: "f32[256]"; primals_12: "f32[256]"; primals_13: "f32[256]"; primals_14: "f32[256]"; primals_15: "f32[64]"; primals_16: "f32[64]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[256]"; primals_20: "f32[256]"; primals_21: "f32[128]"; primals_22: "f32[128]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128]"; primals_32: "f32[128]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[256]"; primals_38: "f32[256]"; primals_39: "f32[1024]"; primals_40: "f32[1024]"; primals_41: "f32[1024]"; primals_42: "f32[1024]"; primals_43: "f32[256]"; primals_44: "f32[256]"; primals_45: "f32[31, 16]"; primals_46: "f32[31, 16]"; primals_47: "f32[256]"; primals_48: "f32[256]"; primals_49: "f32[1024]"; primals_50: "f32[1024]"; primals_51: "f32[512]"; primals_52: "f32[512]"; primals_53: "f32[31, 16]"; primals_54: "f32[31, 16]"; primals_55: "f32[512]"; primals_56: "f32[512]"; primals_57: "f32[2048]"; primals_58: "f32[2048]"; primals_59: "f32[2048]"; primals_60: "f32[2048]"; primals_61: "f32[512]"; primals_62: "f32[512]"; primals_63: "f32[15, 16]"; primals_64: "f32[15, 16]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[2048]"; primals_68: "f32[2048]"; primals_69: "f32[24, 3, 3, 3]"; primals_70: "f32[32, 24, 3, 3]"; primals_71: "f32[64, 32, 3, 3]"; primals_72: "f32[64, 64, 1, 1]"; primals_73: "f32[64, 16, 3, 3]"; primals_74: "f32[1, 1, 3]"; primals_75: "f32[256, 64, 1, 1]"; primals_76: "f32[256, 64, 1, 1]"; primals_77: "f32[64, 256, 1, 1]"; primals_78: "f32[64, 16, 3, 3]"; primals_79: "f32[1, 1, 3]"; primals_80: "f32[256, 64, 1, 1]"; primals_81: "f32[128, 256, 1, 1]"; primals_82: "f32[128, 16, 3, 3]"; primals_83: "f32[1, 1, 5]"; primals_84: "f32[512, 128, 1, 1]"; primals_85: "f32[512, 256, 1, 1]"; primals_86: "f32[128, 512, 1, 1]"; primals_87: "f32[128, 16, 3, 3]"; primals_88: "f32[1, 1, 5]"; primals_89: "f32[512, 128, 1, 1]"; primals_90: "f32[256, 512, 1, 1]"; primals_91: "f32[256, 16, 3, 3]"; primals_92: "f32[1, 1, 5]"; primals_93: "f32[1024, 256, 1, 1]"; primals_94: "f32[1024, 512, 1, 1]"; primals_95: "f32[256, 1024, 1, 1]"; primals_96: "f32[384, 256, 1, 1]"; primals_97: "f32[1024, 256, 1, 1]"; primals_98: "f32[512, 1024, 1, 1]"; primals_99: "f32[640, 512, 1, 1]"; primals_100: "f32[2048, 512, 1, 1]"; primals_101: "f32[2048, 1024, 1, 1]"; primals_102: "f32[512, 2048, 1, 1]"; primals_103: "f32[640, 512, 1, 1]"; primals_104: "f32[2048, 512, 1, 1]"; primals_105: "f32[1000, 2048]"; primals_106: "f32[1000]"; primals_107: "i64[]"; primals_108: "f32[24]"; primals_109: "f32[24]"; primals_110: "i64[]"; primals_111: "f32[32]"; primals_112: "f32[32]"; primals_113: "i64[]"; primals_114: "f32[64]"; primals_115: "f32[64]"; primals_116: "i64[]"; primals_117: "f32[64]"; primals_118: "f32[64]"; primals_119: "i64[]"; primals_120: "f32[64]"; primals_121: "f32[64]"; primals_122: "i64[]"; primals_123: "f32[256]"; primals_124: "f32[256]"; primals_125: "i64[]"; primals_126: "f32[256]"; primals_127: "f32[256]"; primals_128: "i64[]"; primals_129: "f32[64]"; primals_130: "f32[64]"; primals_131: "i64[]"; primals_132: "f32[64]"; primals_133: "f32[64]"; primals_134: "i64[]"; primals_135: "f32[256]"; primals_136: "f32[256]"; primals_137: "i64[]"; primals_138: "f32[128]"; primals_139: "f32[128]"; primals_140: "i64[]"; primals_141: "f32[128]"; primals_142: "f32[128]"; primals_143: "i64[]"; primals_144: "f32[512]"; primals_145: "f32[512]"; primals_146: "i64[]"; primals_147: "f32[512]"; primals_148: "f32[512]"; primals_149: "i64[]"; primals_150: "f32[128]"; primals_151: "f32[128]"; primals_152: "i64[]"; primals_153: "f32[128]"; primals_154: "f32[128]"; primals_155: "i64[]"; primals_156: "f32[512]"; primals_157: "f32[512]"; primals_158: "i64[]"; primals_159: "f32[256]"; primals_160: "f32[256]"; primals_161: "i64[]"; primals_162: "f32[256]"; primals_163: "f32[256]"; primals_164: "i64[]"; primals_165: "f32[1024]"; primals_166: "f32[1024]"; primals_167: "i64[]"; primals_168: "f32[1024]"; primals_169: "f32[1024]"; primals_170: "i64[]"; primals_171: "f32[256]"; primals_172: "f32[256]"; primals_173: "i64[]"; primals_174: "f32[256]"; primals_175: "f32[256]"; primals_176: "i64[]"; primals_177: "f32[1024]"; primals_178: "f32[1024]"; primals_179: "i64[]"; primals_180: "f32[512]"; primals_181: "f32[512]"; primals_182: "i64[]"; primals_183: "f32[512]"; primals_184: "f32[512]"; primals_185: "i64[]"; primals_186: "f32[2048]"; primals_187: "f32[2048]"; primals_188: "i64[]"; primals_189: "f32[2048]"; primals_190: "f32[2048]"; primals_191: "i64[]"; primals_192: "f32[512]"; primals_193: "f32[512]"; primals_194: "i64[]"; primals_195: "f32[512]"; primals_196: "f32[512]"; primals_197: "i64[]"; primals_198: "f32[2048]"; primals_199: "f32[2048]"; primals_200: "f32[8, 3, 256, 256]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
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
    clone: "f32[8, 24, 128, 128]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
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
    clone_1: "f32[8, 32, 128, 128]" = torch.ops.aten.clone.default(add_9)
    sigmoid_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  add_9 = sigmoid_1 = None
    
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
    clone_2: "f32[8, 64, 128, 128]" = torch.ops.aten.clone.default(add_14)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  add_14 = sigmoid_2 = None
    
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
    clone_3: "f32[8, 64, 64, 64]" = torch.ops.aten.clone.default(add_19)
    sigmoid_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_19)
    mul_31: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_3);  add_19 = sigmoid_3 = None
    
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
    clone_4: "f32[8, 64, 64, 64]" = torch.ops.aten.clone.default(add_24)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  add_24 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_39, [2, 3])
    view: "f32[8, 1, 64]" = torch.ops.aten.view.default(mean, [8, 1, -1]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_5: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view, primals_74, None, [1], [1], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_5);  convolution_5 = None
    alias: "f32[8, 1, 64]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_1: "f32[8, 64, 1, 1]" = torch.ops.aten.view.default(sigmoid_5, [8, -1, 1, 1]);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_1, [8, 64, 64, 64]);  view_1 = None
    mul_40: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, expand)
    
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
    clone_5: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_35)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_35)
    mul_55: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_6);  add_35 = sigmoid_6 = None
    
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
    clone_6: "f32[8, 64, 64, 64]" = torch.ops.aten.clone.default(add_40)
    sigmoid_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_40)
    mul_63: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, sigmoid_7);  add_40 = sigmoid_7 = None
    
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
    clone_7: "f32[8, 64, 64, 64]" = torch.ops.aten.clone.default(add_45)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    mul_71: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  add_45 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_1: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_71, [2, 3])
    view_2: "f32[8, 1, 64]" = torch.ops.aten.view.default(mean_1, [8, 1, -1]);  mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_10: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view_2, primals_79, None, [1], [1], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_1: "f32[8, 1, 64]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_3: "f32[8, 64, 1, 1]" = torch.ops.aten.view.default(sigmoid_9, [8, -1, 1, 1]);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_1: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_3, [8, 64, 64, 64]);  view_3 = None
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_71, expand_1)
    
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
    clone_8: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_51)
    sigmoid_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_51)
    mul_80: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_10);  add_51 = sigmoid_10 = None
    
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
    clone_9: "f32[8, 128, 64, 64]" = torch.ops.aten.clone.default(add_56)
    sigmoid_11: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
    mul_88: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_11);  add_56 = sigmoid_11 = None
    
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
    clone_10: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(add_61)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    mul_96: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  add_61 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_2: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_96, [2, 3])
    view_4: "f32[8, 1, 128]" = torch.ops.aten.view.default(mean_2, [8, 1, -1]);  mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_14: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_4, primals_83, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_14);  convolution_14 = None
    alias_2: "f32[8, 1, 128]" = torch.ops.aten.alias.default(sigmoid_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_5: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sigmoid_13, [8, -1, 1, 1]);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_2: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_5, [8, 128, 32, 32]);  view_5 = None
    mul_97: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_96, expand_2)
    
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
    clone_11: "f32[8, 512, 32, 32]" = torch.ops.aten.clone.default(add_72)
    sigmoid_14: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_72)
    mul_112: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, sigmoid_14);  add_72 = sigmoid_14 = None
    
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
    clone_12: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(add_77)
    sigmoid_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_77)
    mul_120: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_77, sigmoid_15);  add_77 = sigmoid_15 = None
    
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
    clone_13: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(add_82)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    mul_128: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  add_82 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_3: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_128, [2, 3])
    view_6: "f32[8, 1, 128]" = torch.ops.aten.view.default(mean_3, [8, 1, -1]);  mean_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_19: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_6, primals_88, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_19);  convolution_19 = None
    alias_3: "f32[8, 1, 128]" = torch.ops.aten.alias.default(sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_7: "f32[8, 128, 1, 1]" = torch.ops.aten.view.default(sigmoid_17, [8, -1, 1, 1]);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_3: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_7, [8, 128, 32, 32]);  view_7 = None
    mul_129: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, expand_3)
    
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
    clone_14: "f32[8, 512, 32, 32]" = torch.ops.aten.clone.default(add_88)
    sigmoid_18: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_88)
    mul_137: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_18);  add_88 = sigmoid_18 = None
    
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
    clone_15: "f32[8, 256, 32, 32]" = torch.ops.aten.clone.default(add_93)
    sigmoid_19: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    mul_145: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_19);  add_93 = sigmoid_19 = None
    
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
    clone_16: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_98)
    sigmoid_20: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_98)
    mul_153: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_98, sigmoid_20);  add_98 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_4: "f32[8, 256]" = torch.ops.aten.mean.dim(mul_153, [2, 3])
    view_8: "f32[8, 1, 256]" = torch.ops.aten.view.default(mean_4, [8, 1, -1]);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_23: "f32[8, 1, 256]" = torch.ops.aten.convolution.default(view_8, primals_92, None, [1], [2], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_21: "f32[8, 1, 256]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_4: "f32[8, 1, 256]" = torch.ops.aten.alias.default(sigmoid_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_9: "f32[8, 256, 1, 1]" = torch.ops.aten.view.default(sigmoid_21, [8, -1, 1, 1]);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_4: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(view_9, [8, 256, 16, 16]);  view_9 = None
    mul_154: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_153, expand_4)
    
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
    clone_17: "f32[8, 1024, 16, 16]" = torch.ops.aten.clone.default(add_109)
    sigmoid_22: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_109)
    mul_169: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_109, sigmoid_22);  add_109 = sigmoid_22 = None
    
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
    clone_18: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_114)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_114)
    mul_177: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_114, sigmoid_23);  add_114 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_27: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_177, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(convolution_27, [64, 64, 256], 1);  convolution_27 = None
    getitem_46: "f32[8, 64, 16, 16]" = split_with_sizes[0]
    getitem_47: "f32[8, 64, 16, 16]" = split_with_sizes[1]
    getitem_48: "f32[8, 256, 16, 16]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_19: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_46, memory_format = torch.contiguous_format);  getitem_46 = None
    view_10: "f32[32, 16, 256]" = torch.ops.aten.view.default(clone_19, [32, 16, 256]);  clone_19 = None
    permute: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_20: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_47, memory_format = torch.contiguous_format);  getitem_47 = None
    view_11: "f32[32, 16, 256]" = torch.ops.aten.view.default(clone_20, [32, 16, 256]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_21: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_48, memory_format = torch.contiguous_format);  getitem_48 = None
    view_12: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_21, [32, 64, 256]);  clone_21 = None
    permute_1: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_5: "f32[32, 256, 16]" = torch.ops.aten.expand.default(permute, [32, 256, 16])
    view_13: "f32[32, 256, 16]" = torch.ops.aten.view.default(expand_5, [32, 256, 16]);  expand_5 = None
    expand_6: "f32[32, 16, 256]" = torch.ops.aten.expand.default(view_11, [32, 16, 256]);  view_11 = None
    view_14: "f32[32, 16, 256]" = torch.ops.aten.view.default(expand_6, [32, 16, 256]);  expand_6 = None
    bmm: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_13, view_14)
    view_15: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm, [32, 256, 256]);  bmm = None
    mul_178: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_15, 0.25);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_16: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(permute, [32, 16, 16, 16]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_2: "f32[16, 31]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_22: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format)
    view_17: "f32[8192, 16]" = torch.ops.aten.view.default(clone_22, [8192, 16]);  clone_22 = None
    mm: "f32[8192, 31]" = torch.ops.aten.mm.default(view_17, permute_2)
    view_18: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm, [32, 16, 16, 31]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_19: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_18, [-1, 16, 31]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_19, [0, 1], 0.0);  view_19 = None
    view_20: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd, [512, 512]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_1: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_20, [0, 15], 0.0);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_21: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_1, [-1, 17, 31]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_1: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_21, 0, 0, 9223372036854775807);  view_21 = None
    slice_2: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 16);  slice_1 = None
    slice_3: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_2, 2, 15, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_22: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_3, [32, 16, 1, 16, 16]);  slice_3 = None
    expand_7: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_22, [-1, -1, 16, -1, -1]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_3: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_7, [0, 1, 3, 2, 4]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_4: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_5: "f32[16, 31]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    clone_23: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_23: "f32[8192, 16]" = torch.ops.aten.view.default(clone_23, [8192, 16]);  clone_23 = None
    mm_1: "f32[8192, 31]" = torch.ops.aten.mm.default(view_23, permute_5)
    view_24: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_1, [32, 16, 16, 31]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_25: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_24, [-1, 16, 31]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_2: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_25, [0, 1], 0.0);  view_25 = None
    view_26: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_2, [512, 512]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_3: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_26, [0, 15], 0.0);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_27: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_3, [-1, 17, 31]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_4: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_27, 0, 0, 9223372036854775807);  view_27 = None
    slice_5: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 16);  slice_4 = None
    slice_6: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_5, 2, 15, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_28: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_6, [32, 16, 1, 16, 16]);  slice_6 = None
    expand_8: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_28, [-1, -1, 16, -1, -1]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_6: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_8, [0, 3, 1, 4, 2]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_115: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_6, permute_3);  permute_6 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_24: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format);  add_115 = None
    view_29: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_24, [32, 256, 256]);  clone_24 = None
    
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
    view_30: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_9, [32, 256, 256]);  expand_9 = None
    expand_10: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_1, [32, 256, 64]);  permute_1 = None
    view_31: "f32[32, 256, 64]" = torch.ops.aten.view.default(expand_10, [32, 256, 64]);  expand_10 = None
    bmm_1: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(view_30, view_31)
    view_32: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_1, [32, 256, 64]);  bmm_1 = None
    permute_7: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    clone_25: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_33: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_25, [8, 256, 16, 16]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(view_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_49: "f32[1, 256, 1, 1]" = var_mean_22[0]
    getitem_50: "f32[1, 256, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05)
    rsqrt_22: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_33, getitem_50)
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
    clone_26: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_121)
    sigmoid_24: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_121)
    mul_186: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_121, sigmoid_24);  add_121 = sigmoid_24 = None
    
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
    clone_27: "f32[8, 1024, 16, 16]" = torch.ops.aten.clone.default(add_127)
    sigmoid_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    mul_194: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_25);  add_127 = sigmoid_25 = None
    
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
    clone_28: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(add_132)
    sigmoid_26: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    mul_202: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sigmoid_26);  add_132 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_30: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_202, primals_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(convolution_30, [64, 64, 512], 1);  convolution_30 = None
    getitem_55: "f32[8, 64, 16, 16]" = split_with_sizes_1[0]
    getitem_56: "f32[8, 64, 16, 16]" = split_with_sizes_1[1]
    getitem_57: "f32[8, 512, 16, 16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_29: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_55, memory_format = torch.contiguous_format);  getitem_55 = None
    view_34: "f32[32, 16, 256]" = torch.ops.aten.view.default(clone_29, [32, 16, 256]);  clone_29 = None
    permute_8: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_30: "f32[8, 64, 16, 16]" = torch.ops.aten.clone.default(getitem_56, memory_format = torch.contiguous_format);  getitem_56 = None
    view_35: "f32[32, 16, 256]" = torch.ops.aten.view.default(clone_30, [32, 16, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_31: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    view_36: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_31, [32, 128, 256]);  clone_31 = None
    permute_9: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_11: "f32[32, 256, 16]" = torch.ops.aten.expand.default(permute_8, [32, 256, 16])
    view_37: "f32[32, 256, 16]" = torch.ops.aten.view.default(expand_11, [32, 256, 16]);  expand_11 = None
    expand_12: "f32[32, 16, 256]" = torch.ops.aten.expand.default(view_35, [32, 16, 256]);  view_35 = None
    view_38: "f32[32, 16, 256]" = torch.ops.aten.view.default(expand_12, [32, 16, 256]);  expand_12 = None
    bmm_2: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_37, view_38)
    view_39: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_2, [32, 256, 256]);  bmm_2 = None
    mul_203: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_39, 0.25);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_40: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(permute_8, [32, 16, 16, 16]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_10: "f32[16, 31]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    clone_32: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(view_40, memory_format = torch.contiguous_format)
    view_41: "f32[8192, 16]" = torch.ops.aten.view.default(clone_32, [8192, 16]);  clone_32 = None
    mm_2: "f32[8192, 31]" = torch.ops.aten.mm.default(view_41, permute_10)
    view_42: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_2, [32, 16, 16, 31]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_43: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_42, [-1, 16, 31]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_4: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_43, [0, 1], 0.0);  view_43 = None
    view_44: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_4, [512, 512]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_5: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_44, [0, 15], 0.0);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_45: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_5, [-1, 17, 31]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_7: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_45, 0, 0, 9223372036854775807);  view_45 = None
    slice_8: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 16);  slice_7 = None
    slice_9: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_8, 2, 15, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_46: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_9, [32, 16, 1, 16, 16]);  slice_9 = None
    expand_13: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_46, [-1, -1, 16, -1, -1]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_11: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_13, [0, 1, 3, 2, 4]);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_12: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_13: "f32[16, 31]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    clone_33: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_47: "f32[8192, 16]" = torch.ops.aten.view.default(clone_33, [8192, 16]);  clone_33 = None
    mm_3: "f32[8192, 31]" = torch.ops.aten.mm.default(view_47, permute_13)
    view_48: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_3, [32, 16, 16, 31]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_49: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_48, [-1, 16, 31]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_6: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_49, [0, 1], 0.0);  view_49 = None
    view_50: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_6, [512, 512]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_7: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_50, [0, 15], 0.0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_51: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_7, [-1, 17, 31]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_10: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_51, 0, 0, 9223372036854775807);  view_51 = None
    slice_11: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 16);  slice_10 = None
    slice_12: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_11, 2, 15, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_52: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_12, [32, 16, 1, 16, 16]);  slice_12 = None
    expand_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_52, [-1, -1, 16, -1, -1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_14, [0, 3, 1, 4, 2]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_133: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_14, permute_11);  permute_14 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_34: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format);  add_133 = None
    view_53: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_34, [32, 256, 256]);  clone_34 = None
    
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
    view_54: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_15, [32, 256, 256]);  expand_15 = None
    expand_16: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_9, [32, 256, 128]);  permute_9 = None
    view_55: "f32[32, 256, 128]" = torch.ops.aten.view.default(expand_16, [32, 256, 128]);  expand_16 = None
    bmm_3: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(view_54, view_55)
    view_56: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_3, [32, 256, 128]);  bmm_3 = None
    permute_15: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    clone_35: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_57: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_35, [8, 512, 16, 16]);  clone_35 = None
    
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
    clone_36: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_139)
    sigmoid_27: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_139)
    mul_211: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_139, sigmoid_27);  add_139 = sigmoid_27 = None
    
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
    clone_37: "f32[8, 2048, 8, 8]" = torch.ops.aten.clone.default(add_150)
    sigmoid_28: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_150)
    mul_226: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_28);  add_150 = sigmoid_28 = None
    
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
    clone_38: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_155)
    sigmoid_29: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_155)
    mul_234: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_29);  add_155 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_234, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(convolution_34, [64, 64, 512], 1);  convolution_34 = None
    getitem_66: "f32[8, 64, 8, 8]" = split_with_sizes_2[0]
    getitem_67: "f32[8, 64, 8, 8]" = split_with_sizes_2[1]
    getitem_68: "f32[8, 512, 8, 8]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_39: "f32[8, 64, 8, 8]" = torch.ops.aten.clone.default(getitem_66, memory_format = torch.contiguous_format);  getitem_66 = None
    view_58: "f32[32, 16, 64]" = torch.ops.aten.view.default(clone_39, [32, 16, 64]);  clone_39 = None
    permute_16: "f32[32, 64, 16]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_40: "f32[8, 64, 8, 8]" = torch.ops.aten.clone.default(getitem_67, memory_format = torch.contiguous_format);  getitem_67 = None
    view_59: "f32[32, 16, 64]" = torch.ops.aten.view.default(clone_40, [32, 16, 64]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_41: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_68, memory_format = torch.contiguous_format);  getitem_68 = None
    view_60: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_41, [32, 128, 64]);  clone_41 = None
    permute_17: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_17: "f32[32, 64, 16]" = torch.ops.aten.expand.default(permute_16, [32, 64, 16])
    view_61: "f32[32, 64, 16]" = torch.ops.aten.view.default(expand_17, [32, 64, 16]);  expand_17 = None
    expand_18: "f32[32, 16, 64]" = torch.ops.aten.expand.default(view_59, [32, 16, 64]);  view_59 = None
    view_62: "f32[32, 16, 64]" = torch.ops.aten.view.default(expand_18, [32, 16, 64]);  expand_18 = None
    bmm_4: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_4, [32, 64, 64]);  bmm_4 = None
    mul_235: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_63, 0.25);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_64: "f32[32, 8, 8, 16]" = torch.ops.aten.view.default(permute_16, [32, 8, 8, 16]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[16, 15]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    clone_42: "f32[32, 8, 8, 16]" = torch.ops.aten.clone.default(view_64, memory_format = torch.contiguous_format)
    view_65: "f32[2048, 16]" = torch.ops.aten.view.default(clone_42, [2048, 16]);  clone_42 = None
    mm_4: "f32[2048, 15]" = torch.ops.aten.mm.default(view_65, permute_18)
    view_66: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_4, [32, 8, 8, 15]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_67: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_66, [-1, 8, 15]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_67, [0, 1], 0.0);  view_67 = None
    view_68: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_8, [256, 128]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_9: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_68, [0, 7], 0.0);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_69: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_9, [-1, 9, 15]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_13: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_69, 0, 0, 9223372036854775807);  view_69 = None
    slice_14: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 8);  slice_13 = None
    slice_15: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_14, 2, 7, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_70: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_15, [32, 8, 1, 8, 8]);  slice_15 = None
    expand_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_70, [-1, -1, 8, -1, -1]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_19: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_19, [0, 1, 3, 2, 4]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_20: "f32[32, 8, 8, 16]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_21: "f32[16, 15]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    clone_43: "f32[32, 8, 8, 16]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_71: "f32[2048, 16]" = torch.ops.aten.view.default(clone_43, [2048, 16]);  clone_43 = None
    mm_5: "f32[2048, 15]" = torch.ops.aten.mm.default(view_71, permute_21)
    view_72: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_5, [32, 8, 8, 15]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_73: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_72, [-1, 8, 15]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_10: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_73, [0, 1], 0.0);  view_73 = None
    view_74: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_10, [256, 128]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_11: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_74, [0, 7], 0.0);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_75: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_11, [-1, 9, 15]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_16: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_75, 0, 0, 9223372036854775807);  view_75 = None
    slice_17: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 8);  slice_16 = None
    slice_18: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_17, 2, 7, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_76: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_18, [32, 8, 1, 8, 8]);  slice_18 = None
    expand_20: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_76, [-1, -1, 8, -1, -1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_22: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_20, [0, 3, 1, 4, 2]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_156: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_22, permute_19);  permute_22 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_44: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_156, memory_format = torch.contiguous_format);  add_156 = None
    view_77: "f32[32, 64, 64]" = torch.ops.aten.view.default(clone_44, [32, 64, 64]);  clone_44 = None
    
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
    view_78: "f32[32, 64, 64]" = torch.ops.aten.view.default(expand_21, [32, 64, 64]);  expand_21 = None
    expand_22: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_17, [32, 64, 128]);  permute_17 = None
    view_79: "f32[32, 64, 128]" = torch.ops.aten.view.default(expand_22, [32, 64, 128]);  expand_22 = None
    bmm_5: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_5, [32, 64, 128]);  bmm_5 = None
    permute_23: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    clone_45: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_81: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_45, [8, 512, 8, 8]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(view_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_69: "f32[1, 512, 1, 1]" = var_mean_29[0]
    getitem_70: "f32[1, 512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_159: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_29: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_32: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_81, getitem_70)
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
    clone_46: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_162)
    sigmoid_30: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_162)
    mul_243: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_162, sigmoid_30);  add_162 = sigmoid_30 = None
    
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
    clone_47: "f32[8, 2048, 8, 8]" = torch.ops.aten.clone.default(add_168)
    sigmoid_31: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_168)
    mul_251: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_168, sigmoid_31);  add_168 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_5: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(mul_251, [-1, -2], True);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_82: "f32[8, 2048]" = torch.ops.aten.view.default(mean_5, [8, 2048]);  mean_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_48: "f32[8, 2048]" = torch.ops.aten.clone.default(view_82);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_24: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_106, clone_48, permute_24);  primals_106 = None
    permute_25: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_6: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
    permute_26: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_7: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_26, clone_48);  permute_26 = clone_48 = None
    permute_27: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_4: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_83: "f32[1000]" = torch.ops.aten.view.default(sum_4, [1000]);  sum_4 = None
    permute_28: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_84: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_23: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_84, [8, 2048, 8, 8]);  view_84 = None
    div_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand_23, 64);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_32: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(clone_47)
    full: "f32[8, 2048, 8, 8]" = torch.ops.aten.full.default([8, 2048, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_34: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(full, sigmoid_32);  full = None
    mul_252: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(clone_47, sub_34);  clone_47 = sub_34 = None
    add_169: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Scalar(mul_252, 1);  mul_252 = None
    mul_253: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_32, add_169);  sigmoid_32 = add_169 = None
    mul_254: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(div_3, mul_253);  div_3 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_124: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_125: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
    unsqueeze_126: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
    sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3])
    sub_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_126)
    mul_255: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_254, sub_35);  sub_35 = None
    sum_6: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
    mul_256: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    unsqueeze_127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
    unsqueeze_128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_257: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    mul_258: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_259: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    unsqueeze_130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_259, 0);  mul_259 = None
    unsqueeze_131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_260: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_67);  primals_67 = None
    unsqueeze_133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
    unsqueeze_134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    sub_36: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_126);  convolution_35 = unsqueeze_126 = None
    mul_261: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_132);  sub_36 = unsqueeze_132 = None
    sub_37: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_254, mul_261);  mul_261 = None
    sub_38: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_37, unsqueeze_129);  sub_37 = unsqueeze_129 = None
    mul_262: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_135);  sub_38 = unsqueeze_135 = None
    mul_263: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_262, mul_243, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = mul_243 = primals_104 = None
    getitem_73: "f32[8, 512, 8, 8]" = convolution_backward[0]
    getitem_74: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_46)
    full_1: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_39: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_33);  full_1 = None
    mul_264: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_46, sub_39);  clone_46 = sub_39 = None
    add_170: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_264, 1);  mul_264 = None
    mul_265: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_33, add_170);  sigmoid_33 = add_170 = None
    mul_266: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_73, mul_265);  getitem_73 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_137: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
    unsqueeze_138: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3])
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_138)
    mul_267: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_266, sub_40);  sub_40 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2, 3]);  mul_267 = None
    mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_140: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    mul_270: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    unsqueeze_142: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_143: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_65);  primals_65 = None
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_146: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    sub_41: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_138);  view_81 = unsqueeze_138 = None
    mul_273: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_144);  sub_41 = unsqueeze_144 = None
    sub_42: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_266, mul_273);  mul_266 = mul_273 = None
    sub_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_141);  sub_42 = unsqueeze_141 = None
    mul_274: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_147);  sub_43 = unsqueeze_147 = None
    mul_275: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_85: "f32[32, 128, 64]" = torch.ops.aten.view.default(mul_274, [32, 128, 64]);  mul_274 = None
    permute_31: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
    view_86: "f32[32, 64, 128]" = torch.ops.aten.view.default(permute_31, [32, 64, 128]);  permute_31 = None
    permute_32: "f32[32, 64, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_6: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(permute_32, view_86);  permute_32 = None
    permute_33: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_7: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_86, permute_33);  view_86 = permute_33 = None
    view_87: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_6, [32, 64, 128]);  bmm_6 = None
    view_88: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_7, [32, 64, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_8: "f32[32, 64, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_276: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_88, alias_8);  view_88 = None
    sum_9: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
    mul_277: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(alias_8, sum_9);  alias_8 = sum_9 = None
    sub_44: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_89: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.view.default(sub_44, [32, 8, 8, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_34: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_89, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_10: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_34, [2], True);  permute_34 = None
    view_90: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_10, [256, 8, 8]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_2: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_2, view_90, 2, 7, 9223372036854775807);  full_2 = view_90 = None
    full_3: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_3, slice_scatter, 1, 0, 8);  full_3 = slice_scatter = None
    full_4: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_1, 0, 0, 9223372036854775807);  full_4 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_91: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_12: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_91, [0, -7]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_92: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_12, [256, 8, 16]);  constant_pad_nd_12 = None
    constant_pad_nd_13: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_92, [0, -1]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_93: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_13, [32, 8, 8, 15]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_94: "f32[2048, 15]" = torch.ops.aten.view.default(view_93, [2048, 15]);  view_93 = None
    permute_35: "f32[15, 2048]" = torch.ops.aten.permute.default(view_94, [1, 0])
    mm_8: "f32[15, 16]" = torch.ops.aten.mm.default(permute_35, view_71);  permute_35 = view_71 = None
    permute_36: "f32[16, 15]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    permute_37: "f32[15, 16]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_9: "f32[2048, 16]" = torch.ops.aten.mm.default(view_94, permute_37);  view_94 = permute_37 = None
    view_95: "f32[32, 8, 8, 16]" = torch.ops.aten.view.default(mm_9, [32, 8, 8, 16]);  mm_9 = None
    permute_38: "f32[15, 16]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_39: "f32[32, 8, 8, 16]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_40: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_89, [0, 1, 3, 2, 4]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_11: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_40, [2], True);  permute_40 = None
    view_96: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_11, [256, 8, 8]);  sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_5: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_3: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_5, view_96, 2, 7, 9223372036854775807);  full_5 = view_96 = None
    full_6: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_3, 1, 0, 8);  full_6 = slice_scatter_3 = None
    full_7: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_4, 0, 0, 9223372036854775807);  full_7 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_97: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_14: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_97, [0, -7]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_98: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_14, [256, 8, 16]);  constant_pad_nd_14 = None
    constant_pad_nd_15: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_98, [0, -1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_99: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_15, [32, 8, 8, 15]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_100: "f32[2048, 15]" = torch.ops.aten.view.default(view_99, [2048, 15]);  view_99 = None
    permute_41: "f32[15, 2048]" = torch.ops.aten.permute.default(view_100, [1, 0])
    mm_10: "f32[15, 16]" = torch.ops.aten.mm.default(permute_41, view_65);  permute_41 = view_65 = None
    permute_42: "f32[16, 15]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    permute_43: "f32[15, 16]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_11: "f32[2048, 16]" = torch.ops.aten.mm.default(view_100, permute_43);  view_100 = permute_43 = None
    view_101: "f32[32, 8, 8, 16]" = torch.ops.aten.view.default(mm_11, [32, 8, 8, 16]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_171: "f32[32, 8, 8, 16]" = torch.ops.aten.add.Tensor(permute_39, view_101);  permute_39 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_44: "f32[15, 16]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_49: "f32[32, 8, 8, 16]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format);  add_171 = None
    view_102: "f32[32, 64, 16]" = torch.ops.aten.view.default(clone_49, [32, 64, 16]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_278: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(sub_44, 0.25);  sub_44 = None
    view_103: "f32[32, 64, 64]" = torch.ops.aten.view.default(mul_278, [32, 64, 64]);  mul_278 = None
    permute_45: "f32[32, 16, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_8: "f32[32, 16, 64]" = torch.ops.aten.bmm.default(permute_45, view_103);  permute_45 = None
    permute_46: "f32[32, 64, 16]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_9: "f32[32, 64, 16]" = torch.ops.aten.bmm.default(view_103, permute_46);  view_103 = permute_46 = None
    view_104: "f32[32, 16, 64]" = torch.ops.aten.view.default(bmm_8, [32, 16, 64]);  bmm_8 = None
    view_105: "f32[32, 64, 16]" = torch.ops.aten.view.default(bmm_9, [32, 64, 16]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_172: "f32[32, 64, 16]" = torch.ops.aten.add.Tensor(view_102, view_105);  view_102 = view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_47: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    clone_50: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_106: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_50, [8, 512, 8, 8]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_107: "f32[8, 64, 8, 8]" = torch.ops.aten.view.default(view_104, [8, 64, 8, 8]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_48: "f32[32, 16, 64]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    clone_51: "f32[32, 16, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    view_108: "f32[8, 64, 8, 8]" = torch.ops.aten.view.default(clone_51, [8, 64, 8, 8]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat: "f32[8, 640, 8, 8]" = torch.ops.aten.cat.default([view_108, view_107, view_106], 1);  view_108 = view_107 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, mul_234, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = mul_234 = primals_103 = None
    getitem_76: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_77: "f32[640, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_38)
    full_8: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_34);  full_8 = None
    mul_279: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_38, sub_45);  clone_38 = sub_45 = None
    add_173: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_279, 1);  mul_279 = None
    mul_280: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_34, add_173);  sigmoid_34 = add_173 = None
    mul_281: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_76, mul_280);  getitem_76 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_149: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
    unsqueeze_150: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3])
    sub_46: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_150)
    mul_282: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_281, sub_46);  sub_46 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_151: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_152: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_286: "f32[512]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_155: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_157: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_158: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    sub_47: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_150);  convolution_33 = unsqueeze_150 = None
    mul_288: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_156);  sub_47 = unsqueeze_156 = None
    sub_48: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_281, mul_288);  mul_281 = mul_288 = None
    sub_49: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_153);  sub_48 = unsqueeze_153 = None
    mul_289: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_159);  sub_49 = unsqueeze_159 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_289, mul_226, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = mul_226 = primals_102 = None
    getitem_79: "f32[8, 2048, 8, 8]" = convolution_backward_2[0]
    getitem_80: "f32[512, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_174: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_254, getitem_79);  mul_254 = getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_35: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(clone_37)
    full_9: "f32[8, 2048, 8, 8]" = torch.ops.aten.full.default([8, 2048, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_50: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_35);  full_9 = None
    mul_291: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(clone_37, sub_50);  clone_37 = sub_50 = None
    add_175: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Scalar(mul_291, 1);  mul_291 = None
    mul_292: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_35, add_175);  sigmoid_35 = add_175 = None
    mul_293: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_174, mul_292);  add_174 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_161: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3])
    sub_51: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_162)
    mul_294: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_293, sub_51);  sub_51 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3]);  mul_294 = None
    mul_295: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
    unsqueeze_164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_296: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_297: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_298: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_299: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    sub_52: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_162);  convolution_32 = unsqueeze_162 = None
    mul_300: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_168);  sub_52 = unsqueeze_168 = None
    sub_53: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_293, mul_300);  mul_300 = None
    sub_54: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_165);  sub_53 = unsqueeze_165 = None
    mul_301: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_171);  sub_54 = unsqueeze_171 = None
    mul_302: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_301, mul_194, primals_101, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_101 = None
    getitem_82: "f32[8, 1024, 16, 16]" = convolution_backward_3[0]
    getitem_83: "f32[2048, 1024, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_172: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_173: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3])
    sub_55: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_174)
    mul_303: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_293, sub_55);  sub_55 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_304: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_175: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_176: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
    mul_305: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_306: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_307: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_308: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_57);  primals_57 = None
    unsqueeze_181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    sub_56: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_174);  convolution_31 = unsqueeze_174 = None
    mul_309: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_180);  sub_56 = unsqueeze_180 = None
    sub_57: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(mul_293, mul_309);  mul_293 = mul_309 = None
    sub_58: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_177);  sub_57 = unsqueeze_177 = None
    mul_310: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_183);  sub_58 = unsqueeze_183 = None
    mul_311: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_310, mul_211, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_211 = primals_100 = None
    getitem_85: "f32[8, 512, 8, 8]" = convolution_backward_4[0]
    getitem_86: "f32[2048, 512, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_36)
    full_10: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_36);  full_10 = None
    mul_312: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_36, sub_59);  clone_36 = sub_59 = None
    add_176: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_176);  sigmoid_36 = add_176 = None
    mul_314: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_85, mul_313);  getitem_85 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_185: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    sub_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186)
    mul_315: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_314, sub_60);  sub_60 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_316: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_187: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_188: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_317: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_318: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_319: "f32[512]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_190: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_191: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_320: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_55);  primals_55 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186);  avg_pool2d = unsqueeze_186 = None
    mul_321: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_192);  sub_61 = unsqueeze_192 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
    sub_63: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_189);  sub_62 = unsqueeze_189 = None
    mul_322: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_195);  sub_63 = unsqueeze_195 = None
    mul_323: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d_backward: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(mul_322, view_57, [2, 2], [2, 2], [0, 0], False, True, None);  mul_322 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_109: "f32[32, 128, 256]" = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
    permute_52: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    view_110: "f32[32, 256, 128]" = torch.ops.aten.view.default(permute_52, [32, 256, 128]);  permute_52 = None
    permute_53: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_10: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(permute_53, view_110);  permute_53 = None
    permute_54: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_11: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_110, permute_54);  view_110 = permute_54 = None
    view_111: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_10, [32, 256, 128]);  bmm_10 = None
    view_112: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_11, [32, 256, 256]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_9: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_324: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_112, alias_9);  view_112 = None
    sum_20: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [-1], True)
    mul_325: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_9, sum_20);  alias_9 = sum_20 = None
    sub_64: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_113: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_64, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_55: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_113, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_21: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_55, [2], True);  permute_55 = None
    view_114: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_21, [512, 16, 16]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_11: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_6: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_11, view_114, 2, 15, 9223372036854775807);  full_11 = view_114 = None
    full_12: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_7: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_12, slice_scatter_6, 1, 0, 16);  full_12 = slice_scatter_6 = None
    full_13: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_8: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_13, slice_scatter_7, 0, 0, 9223372036854775807);  full_13 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_115: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_16: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_115, [0, -15]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_116: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_16, [512, 16, 32]);  constant_pad_nd_16 = None
    constant_pad_nd_17: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_116, [0, -1]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_117: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_17, [32, 16, 16, 31]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_118: "f32[8192, 31]" = torch.ops.aten.view.default(view_117, [8192, 31]);  view_117 = None
    permute_56: "f32[31, 8192]" = torch.ops.aten.permute.default(view_118, [1, 0])
    mm_12: "f32[31, 16]" = torch.ops.aten.mm.default(permute_56, view_47);  permute_56 = view_47 = None
    permute_57: "f32[16, 31]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    permute_58: "f32[31, 16]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_13: "f32[8192, 16]" = torch.ops.aten.mm.default(view_118, permute_58);  view_118 = permute_58 = None
    view_119: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(mm_13, [32, 16, 16, 16]);  mm_13 = None
    permute_59: "f32[31, 16]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_60: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_61: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_113, [0, 1, 3, 2, 4]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_22: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_61, [2], True);  permute_61 = None
    view_120: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_22, [512, 16, 16]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_14: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_9: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_14, view_120, 2, 15, 9223372036854775807);  full_14 = view_120 = None
    full_15: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_10: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_15, slice_scatter_9, 1, 0, 16);  full_15 = slice_scatter_9 = None
    full_16: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_11: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_16, slice_scatter_10, 0, 0, 9223372036854775807);  full_16 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_121: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_18: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_121, [0, -15]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_122: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_18, [512, 16, 32]);  constant_pad_nd_18 = None
    constant_pad_nd_19: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_122, [0, -1]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_123: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_19, [32, 16, 16, 31]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_124: "f32[8192, 31]" = torch.ops.aten.view.default(view_123, [8192, 31]);  view_123 = None
    permute_62: "f32[31, 8192]" = torch.ops.aten.permute.default(view_124, [1, 0])
    mm_14: "f32[31, 16]" = torch.ops.aten.mm.default(permute_62, view_41);  permute_62 = view_41 = None
    permute_63: "f32[16, 31]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    permute_64: "f32[31, 16]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_15: "f32[8192, 16]" = torch.ops.aten.mm.default(view_124, permute_64);  view_124 = permute_64 = None
    view_125: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(mm_15, [32, 16, 16, 16]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_177: "f32[32, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_60, view_125);  permute_60 = view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_65: "f32[31, 16]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_52: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(add_177, memory_format = torch.contiguous_format);  add_177 = None
    view_126: "f32[32, 256, 16]" = torch.ops.aten.view.default(clone_52, [32, 256, 16]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_326: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_64, 0.25);  sub_64 = None
    view_127: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_326, [32, 256, 256]);  mul_326 = None
    permute_66: "f32[32, 16, 256]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_12: "f32[32, 16, 256]" = torch.ops.aten.bmm.default(permute_66, view_127);  permute_66 = None
    permute_67: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
    bmm_13: "f32[32, 256, 16]" = torch.ops.aten.bmm.default(view_127, permute_67);  view_127 = permute_67 = None
    view_128: "f32[32, 16, 256]" = torch.ops.aten.view.default(bmm_12, [32, 16, 256]);  bmm_12 = None
    view_129: "f32[32, 256, 16]" = torch.ops.aten.view.default(bmm_13, [32, 256, 16]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_178: "f32[32, 256, 16]" = torch.ops.aten.add.Tensor(view_126, view_129);  view_126 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_68: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    clone_53: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    view_130: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_53, [8, 512, 16, 16]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_131: "f32[8, 64, 16, 16]" = torch.ops.aten.view.default(view_128, [8, 64, 16, 16]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_69: "f32[32, 16, 256]" = torch.ops.aten.permute.default(add_178, [0, 2, 1]);  add_178 = None
    clone_54: "f32[32, 16, 256]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_132: "f32[8, 64, 16, 16]" = torch.ops.aten.view.default(clone_54, [8, 64, 16, 16]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_1: "f32[8, 640, 16, 16]" = torch.ops.aten.cat.default([view_132, view_131, view_130], 1);  view_132 = view_131 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_1, mul_202, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = mul_202 = primals_99 = None
    getitem_88: "f32[8, 512, 16, 16]" = convolution_backward_5[0]
    getitem_89: "f32[640, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(clone_28)
    full_17: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_65: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_37);  full_17 = None
    mul_327: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(clone_28, sub_65);  clone_28 = sub_65 = None
    add_179: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_327, 1);  mul_327 = None
    mul_328: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_37, add_179);  sigmoid_37 = add_179 = None
    mul_329: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_88, mul_328);  getitem_88 = mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3])
    sub_66: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_198)
    mul_330: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_329, sub_66);  sub_66 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2, 3]);  mul_330 = None
    mul_331: "f32[512]" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
    unsqueeze_199: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_200: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
    mul_333: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_202: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
    unsqueeze_203: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_335: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    sub_67: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_198);  convolution_29 = unsqueeze_198 = None
    mul_336: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_204);  sub_67 = unsqueeze_204 = None
    sub_68: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_329, mul_336);  mul_329 = mul_336 = None
    sub_69: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_201);  sub_68 = unsqueeze_201 = None
    mul_337: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_207);  sub_69 = unsqueeze_207 = None
    mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_337, mul_194, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_337 = mul_194 = primals_98 = None
    getitem_91: "f32[8, 1024, 16, 16]" = convolution_backward_6[0]
    getitem_92: "f32[512, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_180: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_82, getitem_91);  getitem_82 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_38: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(clone_27)
    full_18: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_70: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_38);  full_18 = None
    mul_339: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(clone_27, sub_70);  clone_27 = sub_70 = None
    add_181: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_339, 1);  mul_339 = None
    mul_340: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_38, add_181);  sigmoid_38 = add_181 = None
    mul_341: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_180, mul_340);  add_180 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_209: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3])
    sub_71: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_210)
    mul_342: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_341, sub_71);  sub_71 = None
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
    mul_343: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
    unsqueeze_211: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_212: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_344: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    mul_345: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_346: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    unsqueeze_214: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_215: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_347: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_217: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_218: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_72: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_210);  convolution_28 = unsqueeze_210 = None
    mul_348: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_216);  sub_72 = unsqueeze_216 = None
    sub_73: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_341, mul_348);  mul_348 = None
    sub_74: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_213);  sub_73 = unsqueeze_213 = None
    mul_349: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_219);  sub_74 = unsqueeze_219 = None
    mul_350: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_349, mul_186, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = mul_186 = primals_97 = None
    getitem_94: "f32[8, 256, 16, 16]" = convolution_backward_7[0]
    getitem_95: "f32[1024, 256, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_26)
    full_19: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_75: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_39);  full_19 = None
    mul_351: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_26, sub_75);  clone_26 = sub_75 = None
    add_182: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_351, 1);  mul_351 = None
    mul_352: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_39, add_182);  sigmoid_39 = add_182 = None
    mul_353: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_94, mul_352);  getitem_94 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3])
    sub_76: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_222)
    mul_354: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_353, sub_76);  sub_76 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
    mul_355: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_224: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_356: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_358: "f32[256]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_226: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_227: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_359: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_77: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_222);  view_33 = unsqueeze_222 = None
    mul_360: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_228);  sub_77 = unsqueeze_228 = None
    sub_78: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_353, mul_360);  mul_353 = mul_360 = None
    sub_79: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
    mul_361: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_133: "f32[32, 64, 256]" = torch.ops.aten.view.default(mul_361, [32, 64, 256]);  mul_361 = None
    permute_73: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    view_134: "f32[32, 256, 64]" = torch.ops.aten.view.default(permute_73, [32, 256, 64]);  permute_73 = None
    permute_74: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    bmm_14: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(permute_74, view_134);  permute_74 = None
    permute_75: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_15: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_134, permute_75);  view_134 = permute_75 = None
    view_135: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_14, [32, 256, 64]);  bmm_14 = None
    view_136: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_15, [32, 256, 256]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_10: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_363: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_136, alias_10);  view_136 = None
    sum_29: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
    mul_364: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_10, sum_29);  alias_10 = sum_29 = None
    sub_80: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_137: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_80, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_76: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_137, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_30: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_76, [2], True);  permute_76 = None
    view_138: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_30, [512, 16, 16]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_20: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_12: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_20, view_138, 2, 15, 9223372036854775807);  full_20 = view_138 = None
    full_21: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_13: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_21, slice_scatter_12, 1, 0, 16);  full_21 = slice_scatter_12 = None
    full_22: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_14: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_22, slice_scatter_13, 0, 0, 9223372036854775807);  full_22 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_139: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_20: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_139, [0, -15]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_140: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_140, [0, -1]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_141: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_142: "f32[8192, 31]" = torch.ops.aten.view.default(view_141, [8192, 31]);  view_141 = None
    permute_77: "f32[31, 8192]" = torch.ops.aten.permute.default(view_142, [1, 0])
    mm_16: "f32[31, 16]" = torch.ops.aten.mm.default(permute_77, view_23);  permute_77 = view_23 = None
    permute_78: "f32[16, 31]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    permute_79: "f32[31, 16]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_17: "f32[8192, 16]" = torch.ops.aten.mm.default(view_142, permute_79);  view_142 = permute_79 = None
    view_143: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(mm_17, [32, 16, 16, 16]);  mm_17 = None
    permute_80: "f32[31, 16]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_81: "f32[32, 16, 16, 16]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_82: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_137, [0, 1, 3, 2, 4]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_31: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_82, [2], True);  permute_82 = None
    view_144: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_31, [512, 16, 16]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_23: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_15: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_23, view_144, 2, 15, 9223372036854775807);  full_23 = view_144 = None
    full_24: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_16: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_24, slice_scatter_15, 1, 0, 16);  full_24 = slice_scatter_15 = None
    full_25: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_17: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_25, slice_scatter_16, 0, 0, 9223372036854775807);  full_25 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_145: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_22: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_145, [0, -15]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_146: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_146, [0, -1]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_147: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_148: "f32[8192, 31]" = torch.ops.aten.view.default(view_147, [8192, 31]);  view_147 = None
    permute_83: "f32[31, 8192]" = torch.ops.aten.permute.default(view_148, [1, 0])
    mm_18: "f32[31, 16]" = torch.ops.aten.mm.default(permute_83, view_17);  permute_83 = view_17 = None
    permute_84: "f32[16, 31]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    permute_85: "f32[31, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_19: "f32[8192, 16]" = torch.ops.aten.mm.default(view_148, permute_85);  view_148 = permute_85 = None
    view_149: "f32[32, 16, 16, 16]" = torch.ops.aten.view.default(mm_19, [32, 16, 16, 16]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_183: "f32[32, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_81, view_149);  permute_81 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_86: "f32[31, 16]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_55: "f32[32, 16, 16, 16]" = torch.ops.aten.clone.default(add_183, memory_format = torch.contiguous_format);  add_183 = None
    view_150: "f32[32, 256, 16]" = torch.ops.aten.view.default(clone_55, [32, 256, 16]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_365: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_80, 0.25);  sub_80 = None
    view_151: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_365, [32, 256, 256]);  mul_365 = None
    permute_87: "f32[32, 16, 256]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_16: "f32[32, 16, 256]" = torch.ops.aten.bmm.default(permute_87, view_151);  permute_87 = None
    permute_88: "f32[32, 256, 16]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_17: "f32[32, 256, 16]" = torch.ops.aten.bmm.default(view_151, permute_88);  view_151 = permute_88 = None
    view_152: "f32[32, 16, 256]" = torch.ops.aten.view.default(bmm_16, [32, 16, 256]);  bmm_16 = None
    view_153: "f32[32, 256, 16]" = torch.ops.aten.view.default(bmm_17, [32, 256, 16]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_184: "f32[32, 256, 16]" = torch.ops.aten.add.Tensor(view_150, view_153);  view_150 = view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_89: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    clone_56: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
    view_154: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_56, [8, 256, 16, 16]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_155: "f32[8, 64, 16, 16]" = torch.ops.aten.view.default(view_152, [8, 64, 16, 16]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_90: "f32[32, 16, 256]" = torch.ops.aten.permute.default(add_184, [0, 2, 1]);  add_184 = None
    clone_57: "f32[32, 16, 256]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_156: "f32[8, 64, 16, 16]" = torch.ops.aten.view.default(clone_57, [8, 64, 16, 16]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_2: "f32[8, 384, 16, 16]" = torch.ops.aten.cat.default([view_156, view_155, view_154], 1);  view_156 = view_155 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_2, mul_177, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = mul_177 = primals_96 = None
    getitem_97: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_98: "f32[384, 256, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_18)
    full_26: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_81: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_40);  full_26 = None
    mul_366: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_18, sub_81);  clone_18 = sub_81 = None
    add_185: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_366, 1);  mul_366 = None
    mul_367: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_185);  sigmoid_40 = add_185 = None
    mul_368: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_97, mul_367);  getitem_97 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3])
    sub_82: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_234)
    mul_369: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_368, sub_82);  sub_82 = None
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3]);  mul_369 = None
    mul_370: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_235: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_236: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_371: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_373: "f32[256]" = torch.ops.aten.mul.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
    unsqueeze_239: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_374: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_83: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_234);  convolution_26 = unsqueeze_234 = None
    mul_375: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_240);  sub_83 = unsqueeze_240 = None
    sub_84: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_368, mul_375);  mul_368 = mul_375 = None
    sub_85: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_237);  sub_84 = unsqueeze_237 = None
    mul_376: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_243);  sub_85 = unsqueeze_243 = None
    mul_377: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_376, mul_169, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_376 = mul_169 = primals_95 = None
    getitem_100: "f32[8, 1024, 16, 16]" = convolution_backward_9[0]
    getitem_101: "f32[256, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_186: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_341, getitem_100);  mul_341 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_41: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(clone_17)
    full_27: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_86: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_41);  full_27 = None
    mul_378: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(clone_17, sub_86);  clone_17 = sub_86 = None
    add_187: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_378, 1);  mul_378 = None
    mul_379: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_187);  sigmoid_41 = add_187 = None
    mul_380: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_186, mul_379);  add_186 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_245: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3])
    sub_87: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_246)
    mul_381: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_380, sub_87);  sub_87 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
    mul_382: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_247: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_248: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_383: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_384: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_385: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    unsqueeze_250: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
    unsqueeze_251: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_386: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_253: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_254: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_88: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_246);  convolution_25 = unsqueeze_246 = None
    mul_387: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_252);  sub_88 = unsqueeze_252 = None
    sub_89: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_380, mul_387);  mul_387 = None
    sub_90: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_249);  sub_89 = unsqueeze_249 = None
    mul_388: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_255);  sub_90 = unsqueeze_255 = None
    mul_389: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_388, mul_137, primals_94, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = primals_94 = None
    getitem_103: "f32[8, 512, 32, 32]" = convolution_backward_10[0]
    getitem_104: "f32[1024, 512, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_257: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_36: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3])
    sub_91: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_258)
    mul_390: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_380, sub_91);  sub_91 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
    mul_391: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    unsqueeze_259: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
    unsqueeze_260: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_392: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_393: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_394: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    unsqueeze_262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
    unsqueeze_263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_395: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_92: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_258);  convolution_24 = unsqueeze_258 = None
    mul_396: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_264);  sub_92 = unsqueeze_264 = None
    sub_93: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_380, mul_396);  mul_380 = mul_396 = None
    sub_94: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_261);  sub_93 = unsqueeze_261 = None
    mul_397: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_267);  sub_94 = unsqueeze_267 = None
    mul_398: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_397, mul_154, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = mul_154 = primals_93 = None
    getitem_106: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_107: "f32[1024, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_399: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_106, mul_153);  mul_153 = None
    mul_400: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_106, expand_4);  getitem_106 = expand_4 = None
    sum_38: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2, 3], True);  mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_157: "f32[8, 1, 256]" = torch.ops.aten.view.default(sum_38, [8, 1, 256]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_11: "f32[8, 1, 256]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_95: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(1, alias_11)
    mul_401: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(alias_11, sub_95);  alias_11 = sub_95 = None
    mul_402: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_157, mul_401);  view_157 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_402, view_8, primals_92, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_402 = view_8 = primals_92 = None
    getitem_109: "f32[8, 1, 256]" = convolution_backward_12[0]
    getitem_110: "f32[1, 1, 5]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_158: "f32[8, 256]" = torch.ops.aten.view.default(getitem_109, [8, 256]);  getitem_109 = None
    unsqueeze_268: "f32[8, 256, 1]" = torch.ops.aten.unsqueeze.default(view_158, 2);  view_158 = None
    unsqueeze_269: "f32[8, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    expand_24: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(unsqueeze_269, [8, 256, 16, 16]);  unsqueeze_269 = None
    div_4: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_24, 256);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_188: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_400, div_4);  mul_400 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_16)
    full_28: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_96: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_42);  full_28 = None
    mul_403: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_16, sub_96);  clone_16 = sub_96 = None
    add_189: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_403, 1);  mul_403 = None
    mul_404: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_189);  sigmoid_42 = add_189 = None
    mul_405: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_188, mul_404);  add_188 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_270: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_271: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3])
    sub_97: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_272)
    mul_406: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_405, sub_97);  sub_97 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
    mul_407: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    unsqueeze_273: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
    unsqueeze_274: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_408: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_276: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_277: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_279: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_280: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    sub_98: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_272);  convolution_22 = unsqueeze_272 = None
    mul_412: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_278);  sub_98 = unsqueeze_278 = None
    sub_99: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_405, mul_412);  mul_405 = mul_412 = None
    sub_100: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_275);  sub_99 = unsqueeze_275 = None
    mul_413: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_281);  sub_100 = unsqueeze_281 = None
    mul_414: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_55);  sum_40 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_413, mul_145, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_413 = mul_145 = primals_91 = None
    getitem_112: "f32[8, 256, 32, 32]" = convolution_backward_13[0]
    getitem_113: "f32[256, 16, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(clone_15)
    full_29: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_101: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_43);  full_29 = None
    mul_415: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(clone_15, sub_101);  clone_15 = sub_101 = None
    add_190: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_415, 1);  mul_415 = None
    mul_416: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_190);  sigmoid_43 = add_190 = None
    mul_417: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_112, mul_416);  getitem_112 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_282: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_283: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
    sub_102: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_284)
    mul_418: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_417, sub_102);  sub_102 = None
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
    unsqueeze_285: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_286: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_422: "f32[256]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_288: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_289: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    mul_423: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_291: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_292: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    sub_103: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_284);  convolution_21 = unsqueeze_284 = None
    mul_424: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_290);  sub_103 = unsqueeze_290 = None
    sub_104: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
    sub_105: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_287);  sub_104 = unsqueeze_287 = None
    mul_425: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_293);  sub_105 = unsqueeze_293 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_52);  sum_42 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_425, mul_137, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = mul_137 = primals_90 = None
    getitem_115: "f32[8, 512, 32, 32]" = convolution_backward_14[0]
    getitem_116: "f32[256, 512, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_191: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_103, getitem_115);  getitem_103 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_44: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(clone_14)
    full_30: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_106: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_44);  full_30 = None
    mul_427: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(clone_14, sub_106);  clone_14 = sub_106 = None
    add_192: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_427, 1);  mul_427 = None
    mul_428: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_192);  sigmoid_44 = add_192 = None
    mul_429: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_191, mul_428);  add_191 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_294: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_295: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
    sub_107: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_296)
    mul_430: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_429, sub_107);  sub_107 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
    unsqueeze_297: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_298: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_432: "f32[512]" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    unsqueeze_300: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_301: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_303: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_304: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    sub_108: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_296);  convolution_20 = unsqueeze_296 = None
    mul_436: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_302);  sub_108 = unsqueeze_302 = None
    sub_109: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_436 = None
    sub_110: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_299);  sub_109 = unsqueeze_299 = None
    mul_437: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_305);  sub_110 = unsqueeze_305 = None
    mul_438: "f32[512]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_49);  sum_44 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_437, mul_129, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_129 = primals_89 = None
    getitem_118: "f32[8, 128, 32, 32]" = convolution_backward_15[0]
    getitem_119: "f32[512, 128, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_439: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_118, mul_128);  mul_128 = None
    mul_440: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_118, expand_3);  getitem_118 = expand_3 = None
    sum_45: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2, 3], True);  mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_159: "f32[8, 1, 128]" = torch.ops.aten.view.default(sum_45, [8, 1, 128]);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_12: "f32[8, 1, 128]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_111: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(1, alias_12)
    mul_441: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(alias_12, sub_111);  alias_12 = sub_111 = None
    mul_442: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_159, mul_441);  view_159 = mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_442, view_6, primals_88, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_442 = view_6 = primals_88 = None
    getitem_121: "f32[8, 1, 128]" = convolution_backward_16[0]
    getitem_122: "f32[1, 1, 5]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_160: "f32[8, 128]" = torch.ops.aten.view.default(getitem_121, [8, 128]);  getitem_121 = None
    unsqueeze_306: "f32[8, 128, 1]" = torch.ops.aten.unsqueeze.default(view_160, 2);  view_160 = None
    unsqueeze_307: "f32[8, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    expand_25: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_307, [8, 128, 32, 32]);  unsqueeze_307 = None
    div_5: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_25, 1024);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_193: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_440, div_5);  mul_440 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_13)
    full_31: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_112: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_45);  full_31 = None
    mul_443: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_13, sub_112);  clone_13 = sub_112 = None
    add_194: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_443, 1);  mul_443 = None
    mul_444: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_194);  sigmoid_45 = add_194 = None
    mul_445: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_193, mul_444);  add_193 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3])
    sub_113: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_310)
    mul_446: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_445, sub_113);  sub_113 = None
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
    mul_447: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
    unsqueeze_311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
    unsqueeze_312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_448: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
    mul_449: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_450: "f32[128]" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
    unsqueeze_314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_451: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_114: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_310);  convolution_18 = unsqueeze_310 = None
    mul_452: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_316);  sub_114 = unsqueeze_316 = None
    sub_115: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_445, mul_452);  mul_445 = mul_452 = None
    sub_116: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_313);  sub_115 = unsqueeze_313 = None
    mul_453: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_319);  sub_116 = unsqueeze_319 = None
    mul_454: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_46);  sum_47 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_453, mul_120, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_453 = mul_120 = primals_87 = None
    getitem_124: "f32[8, 128, 32, 32]" = convolution_backward_17[0]
    getitem_125: "f32[128, 16, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_46: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_12)
    full_32: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_117: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_46);  full_32 = None
    mul_455: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_12, sub_117);  clone_12 = sub_117 = None
    add_195: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_455, 1);  mul_455 = None
    mul_456: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_195);  sigmoid_46 = add_195 = None
    mul_457: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_124, mul_456);  getitem_124 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3])
    sub_118: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_322)
    mul_458: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_457, sub_118);  sub_118 = None
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
    unsqueeze_323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_460: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
    mul_461: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_463: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_119: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_322);  convolution_17 = unsqueeze_322 = None
    mul_464: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_328);  sub_119 = unsqueeze_328 = None
    sub_120: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_457, mul_464);  mul_457 = mul_464 = None
    sub_121: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_325);  sub_120 = unsqueeze_325 = None
    mul_465: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_331);  sub_121 = unsqueeze_331 = None
    mul_466: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_43);  sum_49 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_465, mul_112, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_112 = primals_86 = None
    getitem_127: "f32[8, 512, 32, 32]" = convolution_backward_18[0]
    getitem_128: "f32[128, 512, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_196: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_429, getitem_127);  mul_429 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_47: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(clone_11)
    full_33: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_122: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_47);  full_33 = None
    mul_467: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(clone_11, sub_122);  clone_11 = sub_122 = None
    add_197: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_467, 1);  mul_467 = None
    mul_468: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_47, add_197);  sigmoid_47 = add_197 = None
    mul_469: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_196, mul_468);  add_196 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_333: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3])
    sub_123: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_334)
    mul_470: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_469, sub_123);  sub_123 = None
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
    mul_471: "f32[512]" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
    unsqueeze_335: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_336: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_472: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    mul_473: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_474: "f32[512]" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    unsqueeze_338: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_339: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_475: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_341: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_342: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_124: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_334);  convolution_16 = unsqueeze_334 = None
    mul_476: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_340);  sub_124 = unsqueeze_340 = None
    sub_125: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_469, mul_476);  mul_476 = None
    sub_126: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_337);  sub_125 = unsqueeze_337 = None
    mul_477: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_343);  sub_126 = unsqueeze_343 = None
    mul_478: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_40);  sum_51 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_477, mul_80, primals_85, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_85 = None
    getitem_130: "f32[8, 256, 64, 64]" = convolution_backward_19[0]
    getitem_131: "f32[512, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_345: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3])
    sub_127: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_346)
    mul_479: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_469, sub_127);  sub_127 = None
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_480: "f32[512]" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
    unsqueeze_347: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_348: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_481: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    mul_482: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_483: "f32[512]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_350: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_351: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_484: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_353: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_354: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_128: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_346);  convolution_15 = unsqueeze_346 = None
    mul_485: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_352);  sub_128 = unsqueeze_352 = None
    sub_129: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_469, mul_485);  mul_469 = mul_485 = None
    sub_130: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_349);  sub_129 = unsqueeze_349 = None
    mul_486: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_355);  sub_130 = unsqueeze_355 = None
    mul_487: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_37);  sum_53 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_486, mul_97, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_97 = primals_84 = None
    getitem_133: "f32[8, 128, 32, 32]" = convolution_backward_20[0]
    getitem_134: "f32[512, 128, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_488: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_133, mul_96);  mul_96 = None
    mul_489: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_133, expand_2);  getitem_133 = expand_2 = None
    sum_54: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2, 3], True);  mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_161: "f32[8, 1, 128]" = torch.ops.aten.view.default(sum_54, [8, 1, 128]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_13: "f32[8, 1, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    sub_131: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(1, alias_13)
    mul_490: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(alias_13, sub_131);  alias_13 = sub_131 = None
    mul_491: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_161, mul_490);  view_161 = mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_491, view_4, primals_83, [0], [1], [2], [1], False, [0], 1, [True, True, False]);  mul_491 = view_4 = primals_83 = None
    getitem_136: "f32[8, 1, 128]" = convolution_backward_21[0]
    getitem_137: "f32[1, 1, 5]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_162: "f32[8, 128]" = torch.ops.aten.view.default(getitem_136, [8, 128]);  getitem_136 = None
    unsqueeze_356: "f32[8, 128, 1]" = torch.ops.aten.unsqueeze.default(view_162, 2);  view_162 = None
    unsqueeze_357: "f32[8, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    expand_26: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_357, [8, 128, 32, 32]);  unsqueeze_357 = None
    div_6: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_26, 1024);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_198: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_489, div_6);  mul_489 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_10)
    full_34: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_132: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_48);  full_34 = None
    mul_492: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_10, sub_132);  clone_10 = sub_132 = None
    add_199: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_492, 1);  mul_492 = None
    mul_493: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_199);  sigmoid_48 = add_199 = None
    mul_494: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_198, mul_493);  add_198 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_358: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_359: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3])
    sub_133: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_360)
    mul_495: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_494, sub_133);  sub_133 = None
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
    mul_496: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
    unsqueeze_361: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_362: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_497: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
    mul_498: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_499: "f32[128]" = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    unsqueeze_364: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_365: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    mul_500: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_367: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_368: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    sub_134: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_360);  convolution_13 = unsqueeze_360 = None
    mul_501: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_366);  sub_134 = unsqueeze_366 = None
    sub_135: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_494, mul_501);  mul_494 = mul_501 = None
    sub_136: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_363);  sub_135 = unsqueeze_363 = None
    mul_502: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_369);  sub_136 = unsqueeze_369 = None
    mul_503: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_34);  sum_56 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_502, mul_88, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_502 = mul_88 = primals_82 = None
    getitem_139: "f32[8, 128, 64, 64]" = convolution_backward_22[0]
    getitem_140: "f32[128, 16, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(clone_9)
    full_35: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_137: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_49);  full_35 = None
    mul_504: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(clone_9, sub_137);  clone_9 = sub_137 = None
    add_200: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_504, 1);  mul_504 = None
    mul_505: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_200);  sigmoid_49 = add_200 = None
    mul_506: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_139, mul_505);  getitem_139 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_370: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_371: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3])
    sub_138: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_372)
    mul_507: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_506, sub_138);  sub_138 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_508: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
    unsqueeze_373: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_374: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_509: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
    mul_510: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_511: "f32[128]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    unsqueeze_376: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_377: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    mul_512: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_379: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_380: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    sub_139: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_372);  convolution_12 = unsqueeze_372 = None
    mul_513: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_378);  sub_139 = unsqueeze_378 = None
    sub_140: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_506, mul_513);  mul_506 = mul_513 = None
    sub_141: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_375);  sub_140 = unsqueeze_375 = None
    mul_514: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_381);  sub_141 = unsqueeze_381 = None
    mul_515: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_31);  sum_58 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_514, mul_80, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = mul_80 = primals_81 = None
    getitem_142: "f32[8, 256, 64, 64]" = convolution_backward_23[0]
    getitem_143: "f32[128, 256, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_201: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_130, getitem_142);  getitem_130 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_50: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_8)
    full_36: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_142: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_36, sigmoid_50);  full_36 = None
    mul_516: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_8, sub_142);  clone_8 = sub_142 = None
    add_202: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_516, 1);  mul_516 = None
    mul_517: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_202);  sigmoid_50 = add_202 = None
    mul_518: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_201, mul_517);  add_201 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
    sub_143: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_384)
    mul_519: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_518, sub_143);  sub_143 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
    mul_522: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_523: "f32[256]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_388: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_389: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_391: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_392: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    sub_144: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_384);  convolution_11 = unsqueeze_384 = None
    mul_525: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_390);  sub_144 = unsqueeze_390 = None
    sub_145: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_518, mul_525);  mul_525 = None
    sub_146: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_387);  sub_145 = unsqueeze_387 = None
    mul_526: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_393);  sub_146 = unsqueeze_393 = None
    mul_527: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_28);  sum_60 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_526, mul_72, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = mul_72 = primals_80 = None
    getitem_145: "f32[8, 64, 64, 64]" = convolution_backward_24[0]
    getitem_146: "f32[256, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_528: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_145, mul_71);  mul_71 = None
    mul_529: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_145, expand_1);  getitem_145 = expand_1 = None
    sum_61: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_163: "f32[8, 1, 64]" = torch.ops.aten.view.default(sum_61, [8, 1, 64]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_14: "f32[8, 1, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_147: "f32[8, 1, 64]" = torch.ops.aten.sub.Tensor(1, alias_14)
    mul_530: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(alias_14, sub_147);  alias_14 = sub_147 = None
    mul_531: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(view_163, mul_530);  view_163 = mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_531, view_2, primals_79, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_531 = view_2 = primals_79 = None
    getitem_148: "f32[8, 1, 64]" = convolution_backward_25[0]
    getitem_149: "f32[1, 1, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_164: "f32[8, 64]" = torch.ops.aten.view.default(getitem_148, [8, 64]);  getitem_148 = None
    unsqueeze_394: "f32[8, 64, 1]" = torch.ops.aten.unsqueeze.default(view_164, 2);  view_164 = None
    unsqueeze_395: "f32[8, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    expand_27: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_395, [8, 64, 64, 64]);  unsqueeze_395 = None
    div_7: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_27, 4096);  expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_203: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_529, div_7);  mul_529 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_7)
    full_37: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_148: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_37, sigmoid_51);  full_37 = None
    mul_532: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_7, sub_148);  clone_7 = sub_148 = None
    add_204: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_532, 1);  mul_532 = None
    mul_533: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_204);  sigmoid_51 = add_204 = None
    mul_534: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_203, mul_533);  add_203 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_397: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    sum_62: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3])
    sub_149: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_398)
    mul_535: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_534, sub_149);  sub_149 = None
    sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[64]" = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
    unsqueeze_399: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_400: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_537: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
    mul_538: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_539: "f32[64]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_402: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_403: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_540: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_405: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_406: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    sub_150: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_398);  convolution_9 = unsqueeze_398 = None
    mul_541: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_404);  sub_150 = unsqueeze_404 = None
    sub_151: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_534, mul_541);  mul_534 = mul_541 = None
    sub_152: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_401);  sub_151 = unsqueeze_401 = None
    mul_542: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_407);  sub_152 = unsqueeze_407 = None
    mul_543: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_25);  sum_63 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_542, mul_63, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_542 = mul_63 = primals_78 = None
    getitem_151: "f32[8, 64, 64, 64]" = convolution_backward_26[0]
    getitem_152: "f32[64, 16, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_6)
    full_38: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_153: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_38, sigmoid_52);  full_38 = None
    mul_544: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_6, sub_153);  clone_6 = sub_153 = None
    add_205: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_544, 1);  mul_544 = None
    mul_545: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_205);  sigmoid_52 = add_205 = None
    mul_546: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_151, mul_545);  getitem_151 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_409: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    sum_64: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3])
    sub_154: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_410)
    mul_547: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_546, sub_154);  sub_154 = None
    sum_65: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[64]" = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
    unsqueeze_411: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_412: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_549: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
    mul_550: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_551: "f32[64]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_414: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_415: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_552: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_417: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_418: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    sub_155: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_410);  convolution_8 = unsqueeze_410 = None
    mul_553: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_416);  sub_155 = unsqueeze_416 = None
    sub_156: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_546, mul_553);  mul_546 = mul_553 = None
    sub_157: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_413);  sub_156 = unsqueeze_413 = None
    mul_554: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_419);  sub_157 = unsqueeze_419 = None
    mul_555: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_22);  sum_65 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_554, mul_55, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_55 = primals_77 = None
    getitem_154: "f32[8, 256, 64, 64]" = convolution_backward_27[0]
    getitem_155: "f32[64, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_206: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_518, getitem_154);  mul_518 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_53: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_5)
    full_39: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_158: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_39, sigmoid_53);  full_39 = None
    mul_556: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_5, sub_158);  clone_5 = sub_158 = None
    add_207: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_556, 1);  mul_556 = None
    mul_557: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_207);  sigmoid_53 = add_207 = None
    mul_558: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_206, mul_557);  add_206 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_421: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
    sub_159: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_422)
    mul_559: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_558, sub_159);  sub_159 = None
    sum_67: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_560: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
    unsqueeze_423: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_424: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_561: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
    mul_562: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_563: "f32[256]" = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    unsqueeze_426: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_427: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_564: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_429: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_430: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    sub_160: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_422);  convolution_7 = unsqueeze_422 = None
    mul_565: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_428);  sub_160 = unsqueeze_428 = None
    sub_161: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_558, mul_565);  mul_565 = None
    sub_162: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_425);  sub_161 = unsqueeze_425 = None
    mul_566: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_431);  sub_162 = unsqueeze_431 = None
    mul_567: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_19);  sum_67 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_566, getitem_6, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_566 = primals_76 = None
    getitem_157: "f32[8, 64, 64, 64]" = convolution_backward_28[0]
    getitem_158: "f32[256, 64, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_432: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_433: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    sum_68: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
    sub_163: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_434)
    mul_568: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_558, sub_163);  sub_163 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
    mul_569: "f32[256]" = torch.ops.aten.mul.Tensor(sum_68, 3.0517578125e-05)
    unsqueeze_435: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_436: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_570: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
    mul_571: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_572: "f32[256]" = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
    unsqueeze_438: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_439: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_573: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_441: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_442: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    sub_164: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_434);  convolution_6 = unsqueeze_434 = None
    mul_574: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_440);  sub_164 = unsqueeze_440 = None
    sub_165: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_558, mul_574);  mul_558 = mul_574 = None
    sub_166: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_437);  sub_165 = unsqueeze_437 = None
    mul_575: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_443);  sub_166 = unsqueeze_443 = None
    mul_576: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_16);  sum_69 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_575, mul_40, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_575 = mul_40 = primals_75 = None
    getitem_160: "f32[8, 64, 64, 64]" = convolution_backward_29[0]
    getitem_161: "f32[256, 64, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    mul_577: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_160, mul_39);  mul_39 = None
    mul_578: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_160, expand);  getitem_160 = expand = None
    sum_70: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2, 3], True);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    view_165: "f32[8, 1, 64]" = torch.ops.aten.view.default(sum_70, [8, 1, 64]);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_15: "f32[8, 1, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    sub_167: "f32[8, 1, 64]" = torch.ops.aten.sub.Tensor(1, alias_15)
    mul_579: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(alias_15, sub_167);  alias_15 = sub_167 = None
    mul_580: "f32[8, 1, 64]" = torch.ops.aten.mul.Tensor(view_165, mul_579);  view_165 = mul_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_580, view, primals_74, [0], [1], [1], [1], False, [0], 1, [True, True, False]);  mul_580 = view = primals_74 = None
    getitem_163: "f32[8, 1, 64]" = convolution_backward_30[0]
    getitem_164: "f32[1, 1, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    view_166: "f32[8, 64]" = torch.ops.aten.view.default(getitem_163, [8, 64]);  getitem_163 = None
    unsqueeze_444: "f32[8, 64, 1]" = torch.ops.aten.unsqueeze.default(view_166, 2);  view_166 = None
    unsqueeze_445: "f32[8, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    expand_28: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(unsqueeze_445, [8, 64, 64, 64]);  unsqueeze_445 = None
    div_8: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_28, 4096);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    add_208: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_578, div_8);  mul_578 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_54: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_4)
    full_40: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_168: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_40, sigmoid_54);  full_40 = None
    mul_581: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_4, sub_168);  clone_4 = sub_168 = None
    add_209: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_581, 1);  mul_581 = None
    mul_582: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_209);  sigmoid_54 = add_209 = None
    mul_583: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_208, mul_582);  add_208 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_446: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_447: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3])
    sub_169: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_448)
    mul_584: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_583, sub_169);  sub_169 = None
    sum_72: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_585: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 3.0517578125e-05)
    unsqueeze_449: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_450: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_586: "f32[64]" = torch.ops.aten.mul.Tensor(sum_72, 3.0517578125e-05)
    mul_587: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_588: "f32[64]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_452: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_453: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    mul_589: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_455: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_456: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    sub_170: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_448);  convolution_4 = unsqueeze_448 = None
    mul_590: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_454);  sub_170 = unsqueeze_454 = None
    sub_171: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_583, mul_590);  mul_583 = mul_590 = None
    sub_172: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
    mul_591: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_457);  sub_172 = unsqueeze_457 = None
    mul_592: "f32[64]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_13);  sum_72 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_591, mul_31, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_591 = mul_31 = primals_73 = None
    getitem_166: "f32[8, 64, 64, 64]" = convolution_backward_31[0]
    getitem_167: "f32[64, 16, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_55: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_3)
    full_41: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_173: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_41, sigmoid_55);  full_41 = None
    mul_593: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_3, sub_173);  clone_3 = sub_173 = None
    add_210: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_593, 1);  mul_593 = None
    mul_594: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_210);  sigmoid_55 = add_210 = None
    mul_595: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_166, mul_594);  getitem_166 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_458: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_459: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    sum_73: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3])
    sub_174: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_460)
    mul_596: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_595, sub_174);  sub_174 = None
    sum_74: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
    mul_597: "f32[64]" = torch.ops.aten.mul.Tensor(sum_73, 3.0517578125e-05)
    unsqueeze_461: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_462: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_598: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, 3.0517578125e-05)
    mul_599: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_600: "f32[64]" = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
    unsqueeze_464: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_465: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    mul_601: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    sub_175: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_460);  convolution_3 = unsqueeze_460 = None
    mul_602: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_466);  sub_175 = unsqueeze_466 = None
    sub_176: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_595, mul_602);  mul_595 = mul_602 = None
    sub_177: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_463);  sub_176 = unsqueeze_463 = None
    mul_603: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_469);  sub_177 = unsqueeze_469 = None
    mul_604: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_10);  sum_74 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_603, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = getitem_6 = primals_72 = None
    getitem_169: "f32[8, 64, 64, 64]" = convolution_backward_32[0]
    getitem_170: "f32[64, 64, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_211: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_157, getitem_169);  getitem_157 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_211, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7);  add_211 = mul_23 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(clone_2)
    full_42: "f32[8, 64, 128, 128]" = torch.ops.aten.full.default([8, 64, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_178: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_56);  full_42 = None
    mul_605: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(clone_2, sub_178);  clone_2 = sub_178 = None
    add_212: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_212);  sigmoid_56 = add_212 = None
    mul_607: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, mul_606);  max_pool2d_with_indices_backward = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_470: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_471: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    sum_75: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
    sub_179: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_472)
    mul_608: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_607, sub_179);  sub_179 = None
    sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_609: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
    unsqueeze_473: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_474: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_610: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, 7.62939453125e-06)
    mul_611: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_612: "f32[64]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_476: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_477: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    mul_613: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_479: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_480: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    sub_180: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_472);  convolution_2 = unsqueeze_472 = None
    mul_614: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_478);  sub_180 = unsqueeze_478 = None
    sub_181: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_607, mul_614);  mul_607 = mul_614 = None
    sub_182: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
    mul_615: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_481);  sub_182 = unsqueeze_481 = None
    mul_616: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_7);  sum_76 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_615, mul_15, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = mul_15 = primals_71 = None
    getitem_172: "f32[8, 32, 128, 128]" = convolution_backward_33[0]
    getitem_173: "f32[64, 32, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(clone_1)
    full_43: "f32[8, 32, 128, 128]" = torch.ops.aten.full.default([8, 32, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_183: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_57);  full_43 = None
    mul_617: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(clone_1, sub_183);  clone_1 = sub_183 = None
    add_213: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Scalar(mul_617, 1);  mul_617 = None
    mul_618: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_213);  sigmoid_57 = add_213 = None
    mul_619: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_172, mul_618);  getitem_172 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_482: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_483: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    sum_77: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3])
    sub_184: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_484)
    mul_620: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_619, sub_184);  sub_184 = None
    sum_78: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
    mul_621: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, 7.62939453125e-06)
    unsqueeze_485: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_486: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_622: "f32[32]" = torch.ops.aten.mul.Tensor(sum_78, 7.62939453125e-06)
    mul_623: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_624: "f32[32]" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    unsqueeze_488: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_489: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    mul_625: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_491: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_492: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    sub_185: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_484);  convolution_1 = unsqueeze_484 = None
    mul_626: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_490);  sub_185 = unsqueeze_490 = None
    sub_186: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(mul_619, mul_626);  mul_619 = mul_626 = None
    sub_187: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
    mul_627: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_493);  sub_187 = unsqueeze_493 = None
    mul_628: "f32[32]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_4);  sum_78 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_627, mul_7, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_7 = primals_70 = None
    getitem_175: "f32[8, 24, 128, 128]" = convolution_backward_34[0]
    getitem_176: "f32[32, 24, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(clone)
    full_44: "f32[8, 24, 128, 128]" = torch.ops.aten.full.default([8, 24, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_188: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_58);  full_44 = None
    mul_629: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(clone, sub_188);  clone = sub_188 = None
    add_214: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Scalar(mul_629, 1);  mul_629 = None
    mul_630: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_214);  sigmoid_58 = add_214 = None
    mul_631: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_175, mul_630);  getitem_175 = mul_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_494: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_495: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    sum_79: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3])
    sub_189: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_496)
    mul_632: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_631, sub_189);  sub_189 = None
    sum_80: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
    mul_633: "f32[24]" = torch.ops.aten.mul.Tensor(sum_79, 7.62939453125e-06)
    unsqueeze_497: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_498: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_634: "f32[24]" = torch.ops.aten.mul.Tensor(sum_80, 7.62939453125e-06)
    mul_635: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_636: "f32[24]" = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_500: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_501: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    mul_637: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_503: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
    unsqueeze_504: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    sub_190: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_496);  convolution = unsqueeze_496 = None
    mul_638: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_502);  sub_190 = unsqueeze_502 = None
    sub_191: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(mul_631, mul_638);  mul_631 = mul_638 = None
    sub_192: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
    mul_639: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_505);  sub_192 = unsqueeze_505 = None
    mul_640: "f32[24]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_1);  sum_80 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_639, primals_200, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_639 = primals_200 = primals_69 = None
    getitem_179: "f32[24, 3, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
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
    return pytree.tree_unflatten([addmm, mul_640, sum_79, mul_628, sum_77, mul_616, sum_75, mul_604, sum_73, mul_592, sum_71, mul_576, sum_68, mul_567, sum_66, mul_555, sum_64, mul_543, sum_62, mul_527, sum_59, mul_515, sum_57, mul_503, sum_55, mul_487, sum_52, mul_478, sum_50, mul_466, sum_48, mul_454, sum_46, mul_438, sum_43, mul_426, sum_41, mul_414, sum_39, mul_398, sum_36, mul_389, sum_34, mul_377, sum_32, permute_86, permute_80, mul_362, sum_27, mul_350, sum_25, mul_338, sum_23, permute_65, permute_59, mul_323, sum_18, mul_311, sum_16, mul_302, sum_14, mul_290, sum_12, permute_44, permute_38, mul_275, sum_7, mul_263, sum_5, getitem_179, getitem_176, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, getitem_77, getitem_74, permute_28, view_83, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    