from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[24]"; primals_2: "f32[24]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64]"; primals_11: "f32[256]"; primals_12: "f32[256]"; primals_13: "f32[256]"; primals_14: "f32[256]"; primals_15: "f32[64]"; primals_16: "f32[64]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[256]"; primals_20: "f32[256]"; primals_21: "f32[128]"; primals_22: "f32[128]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128]"; primals_32: "f32[128]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[128]"; primals_36: "f32[128]"; primals_37: "f32[63, 32]"; primals_38: "f32[63, 32]"; primals_39: "f32[128]"; primals_40: "f32[128]"; primals_41: "f32[512]"; primals_42: "f32[512]"; primals_43: "f32[256]"; primals_44: "f32[256]"; primals_45: "f32[256]"; primals_46: "f32[256]"; primals_47: "f32[1024]"; primals_48: "f32[1024]"; primals_49: "f32[1024]"; primals_50: "f32[1024]"; primals_51: "f32[256]"; primals_52: "f32[256]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[1024]"; primals_56: "f32[1024]"; primals_57: "f32[256]"; primals_58: "f32[256]"; primals_59: "f32[31, 64]"; primals_60: "f32[31, 64]"; primals_61: "f32[256]"; primals_62: "f32[256]"; primals_63: "f32[1024]"; primals_64: "f32[1024]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[31, 128]"; primals_68: "f32[31, 128]"; primals_69: "f32[512]"; primals_70: "f32[512]"; primals_71: "f32[1536]"; primals_72: "f32[1536]"; primals_73: "f32[1536]"; primals_74: "f32[1536]"; primals_75: "f32[512]"; primals_76: "f32[512]"; primals_77: "f32[15, 128]"; primals_78: "f32[15, 128]"; primals_79: "f32[512]"; primals_80: "f32[512]"; primals_81: "f32[1536]"; primals_82: "f32[1536]"; primals_83: "f32[1280]"; primals_84: "f32[1280]"; primals_85: "f32[24, 3, 3, 3]"; primals_86: "f32[32, 24, 3, 3]"; primals_87: "f32[64, 32, 3, 3]"; primals_88: "f32[64, 64, 1, 1]"; primals_89: "f32[64, 64, 3, 3]"; primals_90: "f32[8, 64, 1, 1]"; primals_91: "f32[8]"; primals_92: "f32[64, 8, 1, 1]"; primals_93: "f32[64]"; primals_94: "f32[256, 64, 1, 1]"; primals_95: "f32[256, 64, 1, 1]"; primals_96: "f32[64, 256, 1, 1]"; primals_97: "f32[64, 64, 3, 3]"; primals_98: "f32[8, 64, 1, 1]"; primals_99: "f32[8]"; primals_100: "f32[64, 8, 1, 1]"; primals_101: "f32[64]"; primals_102: "f32[256, 64, 1, 1]"; primals_103: "f32[128, 256, 1, 1]"; primals_104: "f32[128, 128, 3, 3]"; primals_105: "f32[8, 128, 1, 1]"; primals_106: "f32[8]"; primals_107: "f32[128, 8, 1, 1]"; primals_108: "f32[128]"; primals_109: "f32[512, 128, 1, 1]"; primals_110: "f32[512, 256, 1, 1]"; primals_111: "f32[128, 512, 1, 1]"; primals_112: "f32[128, 128, 3, 3]"; primals_113: "f32[8, 128, 1, 1]"; primals_114: "f32[8]"; primals_115: "f32[128, 8, 1, 1]"; primals_116: "f32[128]"; primals_117: "f32[512, 128, 1, 1]"; primals_118: "f32[128, 512, 1, 1]"; primals_119: "f32[384, 128, 1, 1]"; primals_120: "f32[512, 128, 1, 1]"; primals_121: "f32[256, 512, 1, 1]"; primals_122: "f32[256, 256, 3, 3]"; primals_123: "f32[16, 256, 1, 1]"; primals_124: "f32[16]"; primals_125: "f32[256, 16, 1, 1]"; primals_126: "f32[256]"; primals_127: "f32[1024, 256, 1, 1]"; primals_128: "f32[1024, 512, 1, 1]"; primals_129: "f32[256, 1024, 1, 1]"; primals_130: "f32[256, 256, 3, 3]"; primals_131: "f32[16, 256, 1, 1]"; primals_132: "f32[16]"; primals_133: "f32[256, 16, 1, 1]"; primals_134: "f32[256]"; primals_135: "f32[1024, 256, 1, 1]"; primals_136: "f32[256, 1024, 1, 1]"; primals_137: "f32[768, 256, 1, 1]"; primals_138: "f32[1024, 256, 1, 1]"; primals_139: "f32[512, 1024, 1, 1]"; primals_140: "f32[1536, 512, 1, 1]"; primals_141: "f32[1536, 512, 1, 1]"; primals_142: "f32[1536, 1024, 1, 1]"; primals_143: "f32[512, 1536, 1, 1]"; primals_144: "f32[1536, 512, 1, 1]"; primals_145: "f32[1536, 512, 1, 1]"; primals_146: "f32[1280, 1536, 1, 1]"; primals_147: "f32[1000, 1280]"; primals_148: "f32[1000]"; primals_149: "i64[]"; primals_150: "f32[24]"; primals_151: "f32[24]"; primals_152: "i64[]"; primals_153: "f32[32]"; primals_154: "f32[32]"; primals_155: "i64[]"; primals_156: "f32[64]"; primals_157: "f32[64]"; primals_158: "i64[]"; primals_159: "f32[64]"; primals_160: "f32[64]"; primals_161: "i64[]"; primals_162: "f32[64]"; primals_163: "f32[64]"; primals_164: "i64[]"; primals_165: "f32[256]"; primals_166: "f32[256]"; primals_167: "i64[]"; primals_168: "f32[256]"; primals_169: "f32[256]"; primals_170: "i64[]"; primals_171: "f32[64]"; primals_172: "f32[64]"; primals_173: "i64[]"; primals_174: "f32[64]"; primals_175: "f32[64]"; primals_176: "i64[]"; primals_177: "f32[256]"; primals_178: "f32[256]"; primals_179: "i64[]"; primals_180: "f32[128]"; primals_181: "f32[128]"; primals_182: "i64[]"; primals_183: "f32[128]"; primals_184: "f32[128]"; primals_185: "i64[]"; primals_186: "f32[512]"; primals_187: "f32[512]"; primals_188: "i64[]"; primals_189: "f32[512]"; primals_190: "f32[512]"; primals_191: "i64[]"; primals_192: "f32[128]"; primals_193: "f32[128]"; primals_194: "i64[]"; primals_195: "f32[128]"; primals_196: "f32[128]"; primals_197: "i64[]"; primals_198: "f32[512]"; primals_199: "f32[512]"; primals_200: "i64[]"; primals_201: "f32[128]"; primals_202: "f32[128]"; primals_203: "i64[]"; primals_204: "f32[128]"; primals_205: "f32[128]"; primals_206: "i64[]"; primals_207: "f32[512]"; primals_208: "f32[512]"; primals_209: "i64[]"; primals_210: "f32[256]"; primals_211: "f32[256]"; primals_212: "i64[]"; primals_213: "f32[256]"; primals_214: "f32[256]"; primals_215: "i64[]"; primals_216: "f32[1024]"; primals_217: "f32[1024]"; primals_218: "i64[]"; primals_219: "f32[1024]"; primals_220: "f32[1024]"; primals_221: "i64[]"; primals_222: "f32[256]"; primals_223: "f32[256]"; primals_224: "i64[]"; primals_225: "f32[256]"; primals_226: "f32[256]"; primals_227: "i64[]"; primals_228: "f32[1024]"; primals_229: "f32[1024]"; primals_230: "i64[]"; primals_231: "f32[256]"; primals_232: "f32[256]"; primals_233: "i64[]"; primals_234: "f32[256]"; primals_235: "f32[256]"; primals_236: "i64[]"; primals_237: "f32[1024]"; primals_238: "f32[1024]"; primals_239: "i64[]"; primals_240: "f32[512]"; primals_241: "f32[512]"; primals_242: "i64[]"; primals_243: "f32[512]"; primals_244: "f32[512]"; primals_245: "i64[]"; primals_246: "f32[1536]"; primals_247: "f32[1536]"; primals_248: "i64[]"; primals_249: "f32[1536]"; primals_250: "f32[1536]"; primals_251: "i64[]"; primals_252: "f32[512]"; primals_253: "f32[512]"; primals_254: "i64[]"; primals_255: "f32[512]"; primals_256: "f32[512]"; primals_257: "i64[]"; primals_258: "f32[1536]"; primals_259: "f32[1536]"; primals_260: "i64[]"; primals_261: "f32[1280]"; primals_262: "f32[1280]"; primals_263: "f32[8, 3, 256, 256]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(primals_263, primals_85, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_149, 1)
    
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
    mul_2: "f32[24]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[24]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[24]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
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
    convolution_1: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_7, primals_86, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_152, 1)
    
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
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
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
    convolution_2: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_15, primals_87, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_155, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000030518509476);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 64, 64, 64]" = torch.ops.aten.clone.default(add_14)
    sigmoid_2: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  add_14 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_23, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
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
    convolution_4: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_31, primals_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_161, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_32: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000030518509476);  squeeze_14 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_39, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_5: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_90, primals_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_6: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_6);  convolution_6 = None
    alias_1: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_40: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_40, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
    mul_41: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(mul_44, 0.1);  mul_44 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_47: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_21);  mul_41 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_23);  mul_47 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_23, primals_95, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
    mul_48: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
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
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_55, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_170, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_15)
    mul_56: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
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
    convolution_10: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_63, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_17)
    mul_64: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_71, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_11: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_98, primals_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_1: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_12: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_100, primals_101, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12);  convolution_12 = None
    alias_3: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_71, sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_72, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_176, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 256, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 256, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
    mul_73: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_76: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_77: "f32[256]" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
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
    convolution_14: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_80, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_21)
    mul_81: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_83: "f32[128]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
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
    convolution_15: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_88, primals_104, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_182, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_23)
    mul_89: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001220852154804);  squeeze_35 = None
    mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(mul_92, 0.1);  mul_92 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_96, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_2: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_2, primals_107, primals_108, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    alias_5: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(sigmoid_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_97: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_96, sigmoid_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_97, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
    mul_98: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001220852154804);  squeeze_38 = None
    mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_65: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_104: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_49);  mul_98 = unsqueeze_49 = None
    unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_51);  mul_104 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_80, primals_110, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_188, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
    mul_105: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_106: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_107: "f32[512]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_69: "f32[512]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_108: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
    mul_109: "f32[512]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[512]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
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
    convolution_20: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_112, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_29)
    mul_113: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
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
    convolution_21: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_120, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_31)
    mul_121: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_123: "f32[128]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_125: "f32[128]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[128]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_128, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_22: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_113, primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_23: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_7: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_129: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_129, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
    mul_130: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_131: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_85: "f32[512]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_50: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_133: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_134: "f32[512]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[512]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
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
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_137, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_200, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_35)
    mul_138: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_140: "f32[128]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_91: "f32[128]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_142: "f32[128]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[128]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_92: "f32[128]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_144: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_69);  mul_138 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_71);  mul_144 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(add_93)
    sigmoid_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    mul_145: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_19);  add_93 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_26: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_145, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(convolution_26, [128, 128, 128], 1);  convolution_26 = None
    getitem_36: "f32[8, 128, 32, 32]" = split_with_sizes[0]
    getitem_37: "f32[8, 128, 32, 32]" = split_with_sizes[1]
    getitem_38: "f32[8, 128, 32, 32]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_16: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_36, memory_format = torch.contiguous_format);  getitem_36 = None
    view: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_16, [32, 32, 1024]);  clone_16 = None
    permute: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_17: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_37, memory_format = torch.contiguous_format);  getitem_37 = None
    view_1: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_17, [32, 32, 1024]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_18: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_38, memory_format = torch.contiguous_format);  getitem_38 = None
    view_2: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_18, [32, 32, 1024]);  clone_18 = None
    permute_1: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute, [32, 1024, 32])
    view_3: "f32[32, 1024, 32]" = torch.ops.aten.view.default(expand, [32, 1024, 32]);  expand = None
    expand_1: "f32[32, 32, 1024]" = torch.ops.aten.expand.default(view_1, [32, 32, 1024]);  view_1 = None
    view_4: "f32[32, 32, 1024]" = torch.ops.aten.view.default(expand_1, [32, 32, 1024]);  expand_1 = None
    bmm: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_3, view_4)
    view_5: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(bmm, [32, 1024, 1024]);  bmm = None
    mul_146: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_5, 0.1767766952966369);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_6: "f32[32, 32, 32, 32]" = torch.ops.aten.view.default(permute, [32, 32, 32, 32]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_2: "f32[32, 63]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    clone_19: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(view_6, memory_format = torch.contiguous_format)
    view_7: "f32[32768, 32]" = torch.ops.aten.view.default(clone_19, [32768, 32]);  clone_19 = None
    mm: "f32[32768, 63]" = torch.ops.aten.mm.default(view_7, permute_2)
    view_8: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(mm, [32, 32, 32, 63]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_9: "f32[1024, 32, 63]" = torch.ops.aten.view.default(view_8, [-1, 32, 63]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_9, [0, 1], 0.0);  view_9 = None
    view_10: "f32[1024, 2048]" = torch.ops.aten.view.default(constant_pad_nd, [1024, 2048]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_1: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_10, [0, 31], 0.0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_11: "f32[1024, 33, 63]" = torch.ops.aten.view.default(constant_pad_nd_1, [-1, 33, 63]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_1: "f32[1024, 33, 63]" = torch.ops.aten.slice.Tensor(view_11, 0, 0, 9223372036854775807);  view_11 = None
    slice_2: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 32);  slice_1 = None
    slice_3: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_2, 2, 31, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_12: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.view.default(slice_3, [32, 32, 1, 32, 32]);  slice_3 = None
    expand_2: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_12, [-1, -1, 32, -1, -1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_3: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_2, [0, 1, 3, 2, 4]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_4: "f32[32, 32, 32, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_5: "f32[32, 63]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    clone_20: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_13: "f32[32768, 32]" = torch.ops.aten.view.default(clone_20, [32768, 32]);  clone_20 = None
    mm_1: "f32[32768, 63]" = torch.ops.aten.mm.default(view_13, permute_5)
    view_14: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(mm_1, [32, 32, 32, 63]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_15: "f32[1024, 32, 63]" = torch.ops.aten.view.default(view_14, [-1, 32, 63]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_2: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_15, [0, 1], 0.0);  view_15 = None
    view_16: "f32[1024, 2048]" = torch.ops.aten.view.default(constant_pad_nd_2, [1024, 2048]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_3: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_16, [0, 31], 0.0);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_17: "f32[1024, 33, 63]" = torch.ops.aten.view.default(constant_pad_nd_3, [-1, 33, 63]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_4: "f32[1024, 33, 63]" = torch.ops.aten.slice.Tensor(view_17, 0, 0, 9223372036854775807);  view_17 = None
    slice_5: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 32);  slice_4 = None
    slice_6: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_5, 2, 31, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_18: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.view.default(slice_6, [32, 32, 1, 32, 32]);  slice_6 = None
    expand_3: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_18, [-1, -1, 32, -1, -1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_6: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_3, [0, 3, 1, 4, 2]);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_94: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.add.Tensor(permute_6, permute_3);  permute_6 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_21: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format);  add_94 = None
    view_19: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(clone_21, [32, 1024, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_95: "f32[32, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_146, view_19);  mul_146 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax: "f32[32, 1024, 1]" = torch.ops.aten.amax.default(add_95, [-1], True)
    sub_18: "f32[32, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_95, amax);  add_95 = amax = None
    exp: "f32[32, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_1: "f32[32, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[32, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_8: "f32[32, 1024, 1024]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_4: "f32[32, 1024, 1024]" = torch.ops.aten.expand.default(div, [32, 1024, 1024]);  div = None
    view_20: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_4, [32, 1024, 1024]);  expand_4 = None
    expand_5: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute_1, [32, 1024, 32]);  permute_1 = None
    view_21: "f32[32, 1024, 32]" = torch.ops.aten.view.default(expand_5, [32, 1024, 32]);  expand_5 = None
    bmm_1: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[32, 1024, 32]" = torch.ops.aten.view.default(bmm_1, [32, 1024, 32]);  bmm_1 = None
    permute_7: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    clone_22: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_23: "f32[8, 128, 32, 32]" = torch.ops.aten.view.default(clone_22, [8, 128, 32, 32]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_96: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_39: "f32[1, 128, 1, 1]" = var_mean_18[0]
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_97: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
    rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_23, getitem_40)
    mul_147: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = None
    squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_98: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_99: "f32[128]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_153: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_73);  mul_147 = unsqueeze_73 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_100: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_75);  mul_153 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_23: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(add_100)
    sigmoid_20: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_100)
    mul_154: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_100, sigmoid_20);  add_100 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_154, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_206, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_41: "f32[1, 512, 1, 1]" = var_mean_19[0]
    getitem_42: "f32[1, 512, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_102: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
    rsqrt_19: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_20: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_42)
    mul_155: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = None
    squeeze_57: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    squeeze_58: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_156: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_157: "f32[512]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_103: "f32[512]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_59: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    mul_158: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_159: "f32[512]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[512]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_104: "f32[512]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_76: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_77: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_161: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_77);  mul_155 = unsqueeze_77 = None
    unsqueeze_78: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_79: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_105: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_79);  mul_161 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_106: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_105, mul_137);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    clone_24: "f32[8, 512, 32, 32]" = torch.ops.aten.clone.default(add_106)
    sigmoid_21: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_106)
    mul_162: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_106, sigmoid_21);  add_106 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_162, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_43: "f32[1, 256, 1, 1]" = var_mean_20[0]
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_108: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05)
    rsqrt_20: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_21: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_44)
    mul_163: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = None
    squeeze_60: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    squeeze_61: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_164: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_109: "f32[256]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    squeeze_62: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    mul_166: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_167: "f32[256]" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
    mul_168: "f32[256]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_110: "f32[256]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_169: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_81);  mul_163 = unsqueeze_81 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_111: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_83);  mul_169 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 256, 32, 32]" = torch.ops.aten.clone.default(add_111)
    sigmoid_22: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_111)
    mul_170: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_111, sigmoid_22);  add_111 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_170, primals_122, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_212, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_21[0]
    getitem_46: "f32[1, 256, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_113: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05)
    rsqrt_21: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_22: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_46)
    mul_171: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = None
    squeeze_63: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    squeeze_64: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_114: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_175: "f32[256]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_115: "f32[256]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_177: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_85);  mul_171 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_116: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_87);  mul_177 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_26: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_116)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_116)
    mul_178: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_23);  add_116 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_178, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_30: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_123, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_4: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_31: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_4, primals_125, primals_126, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_24: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31);  convolution_31 = None
    alias_10: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(sigmoid_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_179: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_178, sigmoid_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_179, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_47: "f32[1, 1024, 1, 1]" = var_mean_22[0]
    getitem_48: "f32[1, 1024, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-05)
    rsqrt_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_48)
    mul_180: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
    squeeze_66: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    squeeze_67: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_181: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_182: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_119: "f32[1024]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    squeeze_68: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_183, 0.1);  mul_183 = None
    mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_120: "f32[1024]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    unsqueeze_88: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_89: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_186: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_89);  mul_180 = unsqueeze_89 = None
    unsqueeze_90: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_91: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_91);  mul_186 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_162, primals_128, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_218, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_49: "f32[1, 1024, 1, 1]" = var_mean_23[0]
    getitem_50: "f32[1, 1024, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05)
    rsqrt_23: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_50)
    mul_187: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
    squeeze_69: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    squeeze_70: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_188: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_189: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_124: "f32[1024]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    squeeze_71: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_190, 0.1);  mul_190 = None
    mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_125: "f32[1024]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    unsqueeze_92: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_93: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_193: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_93);  mul_187 = unsqueeze_93 = None
    unsqueeze_94: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_95: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_95);  mul_193 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_127: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_121, add_126);  add_121 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    clone_27: "f32[8, 1024, 16, 16]" = torch.ops.aten.clone.default(add_127)
    sigmoid_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    mul_194: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_25);  add_127 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_194, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_221, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_51: "f32[1, 256, 1, 1]" = var_mean_24[0]
    getitem_52: "f32[1, 256, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-05)
    rsqrt_24: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_52)
    mul_195: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_130: "f32[256]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(mul_198, 0.1);  mul_198 = None
    mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_131: "f32[256]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_201: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_97);  mul_195 = unsqueeze_97 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_99);  mul_201 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_132)
    sigmoid_26: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    mul_202: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sigmoid_26);  add_132 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_202, primals_130, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_224, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_53: "f32[1, 256, 1, 1]" = var_mean_25[0]
    getitem_54: "f32[1, 256, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05)
    rsqrt_25: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_26: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_54)
    mul_203: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_25);  sub_26 = None
    squeeze_75: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    squeeze_76: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_135: "f32[256]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_77: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
    mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_136: "f32[256]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_209: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_137: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_29: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_137)
    sigmoid_27: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_137)
    mul_210: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_27);  add_137 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_210, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_36: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_5: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_37: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_133, primals_134, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_28: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37);  convolution_37 = None
    alias_12: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(sigmoid_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_211: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_210, sigmoid_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_211, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_227, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_55: "f32[1, 1024, 1, 1]" = var_mean_26[0]
    getitem_56: "f32[1, 1024, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_139: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05)
    rsqrt_26: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_27: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_56)
    mul_212: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_26);  sub_27 = None
    squeeze_78: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    squeeze_79: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_213: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_140: "f32[1024]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_80: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0004885197850513);  squeeze_80 = None
    mul_216: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_141: "f32[1024]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_218: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_105);  mul_212 = unsqueeze_105 = None
    unsqueeze_106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_142: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_107);  mul_218 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_143: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_142, mul_194);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    clone_30: "f32[8, 1024, 16, 16]" = torch.ops.aten.clone.default(add_143)
    sigmoid_29: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_143)
    mul_219: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_143, sigmoid_29);  add_143 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_219, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_230, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_57: "f32[1, 256, 1, 1]" = var_mean_27[0]
    getitem_58: "f32[1, 256, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_145: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05)
    rsqrt_27: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_28: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_58)
    mul_220: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_27);  sub_28 = None
    squeeze_81: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    squeeze_82: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_146: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    squeeze_83: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0004885197850513);  squeeze_83 = None
    mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
    mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_147: "f32[256]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_226: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_109);  mul_220 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_148: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_111);  mul_226 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_148)
    sigmoid_30: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_148)
    mul_227: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_30);  add_148 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_40: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_227, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(convolution_40, [256, 256, 256], 1);  convolution_40 = None
    getitem_59: "f32[8, 256, 16, 16]" = split_with_sizes_1[0]
    getitem_60: "f32[8, 256, 16, 16]" = split_with_sizes_1[1]
    getitem_61: "f32[8, 256, 16, 16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_32: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_59, memory_format = torch.contiguous_format);  getitem_59 = None
    view_24: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_32, [32, 64, 256]);  clone_32 = None
    permute_8: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_33: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_60, memory_format = torch.contiguous_format);  getitem_60 = None
    view_25: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_33, [32, 64, 256]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_34: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_61, memory_format = torch.contiguous_format);  getitem_61 = None
    view_26: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_34, [32, 64, 256]);  clone_34 = None
    permute_9: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_6: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_8, [32, 256, 64])
    view_27: "f32[32, 256, 64]" = torch.ops.aten.view.default(expand_6, [32, 256, 64]);  expand_6 = None
    expand_7: "f32[32, 64, 256]" = torch.ops.aten.expand.default(view_25, [32, 64, 256]);  view_25 = None
    view_28: "f32[32, 64, 256]" = torch.ops.aten.view.default(expand_7, [32, 64, 256]);  expand_7 = None
    bmm_2: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_27, view_28)
    view_29: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_2, [32, 256, 256]);  bmm_2 = None
    mul_228: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_29, 0.125);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_30: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(permute_8, [32, 16, 16, 64]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_10: "f32[64, 31]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    clone_35: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(view_30, memory_format = torch.contiguous_format)
    view_31: "f32[8192, 64]" = torch.ops.aten.view.default(clone_35, [8192, 64]);  clone_35 = None
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
    permute_12: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_13: "f32[64, 31]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    clone_36: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_37: "f32[8192, 64]" = torch.ops.aten.view.default(clone_36, [8192, 64]);  clone_36 = None
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
    add_149: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_14, permute_11);  permute_14 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_37: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format);  add_149 = None
    view_43: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_37, [32, 256, 256]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_150: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_228, view_43);  mul_228 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
    sub_29: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_150, amax_1);  add_150 = amax_1 = None
    exp_1: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_13: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_10: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_1, [32, 256, 256]);  div_1 = None
    view_44: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_10, [32, 256, 256]);  expand_10 = None
    expand_11: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_9, [32, 256, 64]);  permute_9 = None
    view_45: "f32[32, 256, 64]" = torch.ops.aten.view.default(expand_11, [32, 256, 64]);  expand_11 = None
    bmm_3: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_3, [32, 256, 64]);  bmm_3 = None
    permute_15: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    clone_38: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_47: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_38, [8, 256, 16, 16]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_233, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 256, 1, 1]" = var_mean_28[0]
    getitem_63: "f32[1, 256, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_28: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_30: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_47, getitem_63)
    mul_229: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_230: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_231: "f32[256]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_153: "f32[256]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_232: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0004885197850513);  squeeze_86 = None
    mul_233: "f32[256]" = torch.ops.aten.mul.Tensor(mul_232, 0.1);  mul_232 = None
    mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_154: "f32[256]" = torch.ops.aten.add.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_235: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_113);  mul_229 = unsqueeze_113 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_115);  mul_235 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(add_155)
    sigmoid_31: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_155)
    mul_236: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_31);  add_155 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_236, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_236, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 1024, 1, 1]" = var_mean_29[0]
    getitem_65: "f32[1, 1024, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_157: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_29: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_31: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_65)
    mul_237: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_29);  sub_31 = None
    squeeze_87: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_88: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_238: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_239: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_158: "f32[1024]" = torch.ops.aten.add.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
    squeeze_89: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_240: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0004885197850513);  squeeze_89 = None
    mul_241: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_240, 0.1);  mul_240 = None
    mul_242: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_159: "f32[1024]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    unsqueeze_116: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_117: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_243: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_117);  mul_237 = unsqueeze_117 = None
    unsqueeze_118: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_119: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_160: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_119);  mul_243 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_161: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_160, mul_219);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    clone_40: "f32[8, 1024, 16, 16]" = torch.ops.aten.clone.default(add_161)
    sigmoid_32: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_161)
    mul_244: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_32);  add_161 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_244, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 512, 1, 1]" = var_mean_30[0]
    getitem_67: "f32[1, 512, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_163: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_30: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_32: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_67)
    mul_245: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_30);  sub_32 = None
    squeeze_90: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_91: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_164: "f32[512]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_92: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0004885197850513);  squeeze_92 = None
    mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[512]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_165: "f32[512]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_120: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_121: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_251: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_121);  mul_245 = unsqueeze_121 = None
    unsqueeze_122: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_123: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_166: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_123);  mul_251 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_41: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(add_166)
    sigmoid_33: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
    mul_252: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_33);  add_166 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_43: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_252, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(convolution_43, [512, 512, 512], 1);  convolution_43 = None
    getitem_68: "f32[8, 512, 16, 16]" = split_with_sizes_2[0]
    getitem_69: "f32[8, 512, 16, 16]" = split_with_sizes_2[1]
    getitem_70: "f32[8, 512, 16, 16]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_42: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_68, memory_format = torch.contiguous_format);  getitem_68 = None
    view_48: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_42, [32, 128, 256]);  clone_42 = None
    permute_16: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_43: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format);  getitem_69 = None
    view_49: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_43, [32, 128, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_44: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_70, memory_format = torch.contiguous_format);  getitem_70 = None
    view_50: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_44, [32, 128, 256]);  clone_44 = None
    permute_17: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_12: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_16, [32, 256, 128])
    view_51: "f32[32, 256, 128]" = torch.ops.aten.view.default(expand_12, [32, 256, 128]);  expand_12 = None
    expand_13: "f32[32, 128, 256]" = torch.ops.aten.expand.default(view_49, [32, 128, 256]);  view_49 = None
    view_52: "f32[32, 128, 256]" = torch.ops.aten.view.default(expand_13, [32, 128, 256]);  expand_13 = None
    bmm_4: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_51, view_52)
    view_53: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_4, [32, 256, 256]);  bmm_4 = None
    mul_253: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_53, 0.08838834764831845);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_54: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(permute_16, [32, 16, 16, 128]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[128, 31]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    clone_45: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(view_54, memory_format = torch.contiguous_format)
    view_55: "f32[8192, 128]" = torch.ops.aten.view.default(clone_45, [8192, 128]);  clone_45 = None
    mm_4: "f32[8192, 31]" = torch.ops.aten.mm.default(view_55, permute_18)
    view_56: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_4, [32, 16, 16, 31]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_57: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_56, [-1, 16, 31]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 1], 0.0);  view_57 = None
    view_58: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_8, [512, 512]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_9: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_58, [0, 15], 0.0);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_59: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_9, [-1, 17, 31]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_13: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_59, 0, 0, 9223372036854775807);  view_59 = None
    slice_14: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 16);  slice_13 = None
    slice_15: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_14, 2, 15, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_60: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_15, [32, 16, 1, 16, 16]);  slice_15 = None
    expand_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_60, [-1, -1, 16, -1, -1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_19: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_14, [0, 1, 3, 2, 4]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_20: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_21: "f32[128, 31]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    clone_46: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_61: "f32[8192, 128]" = torch.ops.aten.view.default(clone_46, [8192, 128]);  clone_46 = None
    mm_5: "f32[8192, 31]" = torch.ops.aten.mm.default(view_61, permute_21)
    view_62: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_5, [32, 16, 16, 31]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_63: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_62, [-1, 16, 31]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_10: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_63, [0, 1], 0.0);  view_63 = None
    view_64: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_10, [512, 512]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_11: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_64, [0, 15], 0.0);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_65: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_11, [-1, 17, 31]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_16: "f32[512, 17, 31]" = torch.ops.aten.slice.Tensor(view_65, 0, 0, 9223372036854775807);  view_65 = None
    slice_17: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 16);  slice_16 = None
    slice_18: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_17, 2, 15, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_66: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_18, [32, 16, 1, 16, 16]);  slice_18 = None
    expand_15: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_66, [-1, -1, 16, -1, -1]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_22: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_15, [0, 3, 1, 4, 2]);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_167: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_22, permute_19);  permute_22 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_47: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    view_67: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_47, [32, 256, 256]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_168: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_253, view_67);  mul_253 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_168, [-1], True)
    sub_33: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_168, amax_2);  add_168 = amax_2 = None
    exp_2: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_3: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_14: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_16: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_2, [32, 256, 256]);  div_2 = None
    view_68: "f32[32, 256, 256]" = torch.ops.aten.view.default(expand_16, [32, 256, 256]);  expand_16 = None
    expand_17: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_17, [32, 256, 128]);  permute_17 = None
    view_69: "f32[32, 256, 128]" = torch.ops.aten.view.default(expand_17, [32, 256, 128]);  expand_17 = None
    bmm_5: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(view_68, view_69)
    view_70: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_5, [32, 256, 128]);  bmm_5 = None
    permute_23: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    clone_48: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_71: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_48, [8, 512, 16, 16]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_71, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_242, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(avg_pool2d, [0, 2, 3], correction = 0, keepdim = True)
    getitem_71: "f32[1, 512, 1, 1]" = var_mean_31[0]
    getitem_72: "f32[1, 512, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_170: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_31: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_72)
    mul_254: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_31);  sub_34 = None
    squeeze_93: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    squeeze_94: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_255: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_171: "f32[512]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    squeeze_95: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_257: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
    mul_258: "f32[512]" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
    mul_259: "f32[512]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_172: "f32[512]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    unsqueeze_124: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_125: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_260: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_125);  mul_254 = unsqueeze_125 = None
    unsqueeze_126: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_127: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_173: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_127);  mul_260 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_49: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_173)
    sigmoid_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_173)
    mul_261: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_173, sigmoid_34);  add_173 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_261, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_73: "f32[1, 1536, 1, 1]" = var_mean_32[0]
    getitem_74: "f32[1, 1536, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_175: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05)
    rsqrt_32: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_35: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_74)
    mul_262: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_32);  sub_35 = None
    squeeze_96: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    squeeze_97: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_263: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_264: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_176: "f32[1536]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    squeeze_98: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    mul_265: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0019569471624266);  squeeze_98 = None
    mul_266: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_265, 0.1);  mul_265 = None
    mul_267: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_177: "f32[1536]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_128: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_129: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_268: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_129);  mul_262 = unsqueeze_129 = None
    unsqueeze_130: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_131: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_178: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_131);  mul_268 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_244, primals_142, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_248, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_75: "f32[1, 1536, 1, 1]" = var_mean_33[0]
    getitem_76: "f32[1, 1536, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_180: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05)
    rsqrt_33: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_36: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_76)
    mul_269: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_33);  sub_36 = None
    squeeze_99: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    squeeze_100: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_270: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_271: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_181: "f32[1536]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    squeeze_101: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    mul_272: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0019569471624266);  squeeze_101 = None
    mul_273: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_272, 0.1);  mul_272 = None
    mul_274: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_182: "f32[1536]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    unsqueeze_132: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_133: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_275: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_133);  mul_269 = unsqueeze_133 = None
    unsqueeze_134: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_135: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_183: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_135);  mul_275 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_184: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_178, add_183);  add_178 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    clone_50: "f32[8, 1536, 8, 8]" = torch.ops.aten.clone.default(add_184)
    sigmoid_35: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_184)
    mul_276: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_184, sigmoid_35);  add_184 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_276, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_251, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_77: "f32[1, 512, 1, 1]" = var_mean_34[0]
    getitem_78: "f32[1, 512, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_186: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05)
    rsqrt_34: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_37: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_78)
    mul_277: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_34);  sub_37 = None
    squeeze_102: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    squeeze_103: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_278: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_279: "f32[512]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_187: "f32[512]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    squeeze_104: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    mul_280: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0019569471624266);  squeeze_104 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(mul_280, 0.1);  mul_280 = None
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_283: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_137);  mul_277 = unsqueeze_137 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_189: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_139);  mul_283 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_51: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_189)
    sigmoid_36: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_189)
    mul_284: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_189, sigmoid_36);  add_189 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_47: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_284, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(convolution_47, [512, 512, 512], 1);  convolution_47 = None
    getitem_79: "f32[8, 512, 8, 8]" = split_with_sizes_3[0]
    getitem_80: "f32[8, 512, 8, 8]" = split_with_sizes_3[1]
    getitem_81: "f32[8, 512, 8, 8]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_52: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_79, memory_format = torch.contiguous_format);  getitem_79 = None
    view_72: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_52, [32, 128, 64]);  clone_52 = None
    permute_24: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_53: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_80, memory_format = torch.contiguous_format);  getitem_80 = None
    view_73: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_53, [32, 128, 64]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_54: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    view_74: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_54, [32, 128, 64]);  clone_54 = None
    permute_25: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_18: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_24, [32, 64, 128])
    view_75: "f32[32, 64, 128]" = torch.ops.aten.view.default(expand_18, [32, 64, 128]);  expand_18 = None
    expand_19: "f32[32, 128, 64]" = torch.ops.aten.expand.default(view_73, [32, 128, 64]);  view_73 = None
    view_76: "f32[32, 128, 64]" = torch.ops.aten.view.default(expand_19, [32, 128, 64]);  expand_19 = None
    bmm_6: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_6, [32, 64, 64]);  bmm_6 = None
    mul_285: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_77, 0.08838834764831845);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_78: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(permute_24, [32, 8, 8, 128]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_26: "f32[128, 15]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    clone_55: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(view_78, memory_format = torch.contiguous_format)
    view_79: "f32[2048, 128]" = torch.ops.aten.view.default(clone_55, [2048, 128]);  clone_55 = None
    mm_6: "f32[2048, 15]" = torch.ops.aten.mm.default(view_79, permute_26)
    view_80: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_6, [32, 8, 8, 15]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_81: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_80, [-1, 8, 15]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_12: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_81, [0, 1], 0.0);  view_81 = None
    view_82: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_12, [256, 128]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_13: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_82, [0, 7], 0.0);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_83: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_13, [-1, 9, 15]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_19: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_83, 0, 0, 9223372036854775807);  view_83 = None
    slice_20: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 8);  slice_19 = None
    slice_21: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_20, 2, 7, 9223372036854775807);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_84: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_21, [32, 8, 1, 8, 8]);  slice_21 = None
    expand_20: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_84, [-1, -1, 8, -1, -1]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_27: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_20, [0, 1, 3, 2, 4]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_28: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_29: "f32[128, 15]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    clone_56: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_85: "f32[2048, 128]" = torch.ops.aten.view.default(clone_56, [2048, 128]);  clone_56 = None
    mm_7: "f32[2048, 15]" = torch.ops.aten.mm.default(view_85, permute_29)
    view_86: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_7, [32, 8, 8, 15]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_87: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_86, [-1, 8, 15]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_14: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_87, [0, 1], 0.0);  view_87 = None
    view_88: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_14, [256, 128]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_15: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_88, [0, 7], 0.0);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_89: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_15, [-1, 9, 15]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_22: "f32[256, 9, 15]" = torch.ops.aten.slice.Tensor(view_89, 0, 0, 9223372036854775807);  view_89 = None
    slice_23: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(slice_22, 1, 0, 8);  slice_22 = None
    slice_24: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_23, 2, 7, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_90: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_24, [32, 8, 1, 8, 8]);  slice_24 = None
    expand_21: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_90, [-1, -1, 8, -1, -1]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_30: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_21, [0, 3, 1, 4, 2]);  expand_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_190: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_30, permute_27);  permute_30 = permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_57: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_190, memory_format = torch.contiguous_format);  add_190 = None
    view_91: "f32[32, 64, 64]" = torch.ops.aten.view.default(clone_57, [32, 64, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_191: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_285, view_91);  mul_285 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_191, [-1], True)
    sub_38: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_191, amax_3);  add_191 = amax_3 = None
    exp_3: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_4: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_15: "f32[32, 64, 64]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_22: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_3, [32, 64, 64]);  div_3 = None
    view_92: "f32[32, 64, 64]" = torch.ops.aten.view.default(expand_22, [32, 64, 64]);  expand_22 = None
    expand_23: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_25, [32, 64, 128]);  permute_25 = None
    view_93: "f32[32, 64, 128]" = torch.ops.aten.view.default(expand_23, [32, 64, 128]);  expand_23 = None
    bmm_7: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(view_92, view_93)
    view_94: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_7, [32, 64, 128]);  bmm_7 = None
    permute_31: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    clone_58: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_95: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_58, [8, 512, 8, 8]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1, 1]" = var_mean_35[0]
    getitem_83: "f32[1, 512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_193: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_35: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_39: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_95, getitem_83)
    mul_286: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_35);  sub_39 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_194: "f32[512]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0019569471624266);  squeeze_107 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_195: "f32[512]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_292: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_141);  mul_286 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_196: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_143);  mul_292 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_59: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_196)
    sigmoid_37: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_196)
    mul_293: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_196, sigmoid_37);  add_196 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_293, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1, 1]" = var_mean_36[0]
    getitem_85: "f32[1, 1536, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_198: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_36: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_40: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_85)
    mul_294: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_36);  sub_40 = None
    squeeze_108: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_109: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_295: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_296: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_199: "f32[1536]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0019569471624266);  squeeze_110 = None
    mul_298: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_200: "f32[1536]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_144: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_145: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_300: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_145);  mul_294 = unsqueeze_145 = None
    unsqueeze_146: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_147: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_201: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_147);  mul_300 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_202: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_201, mul_276);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    clone_60: "f32[8, 1536, 8, 8]" = torch.ops.aten.clone.default(add_202)
    sigmoid_38: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_202)
    mul_301: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_202, sigmoid_38);  add_202 = sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_301, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1280, 1, 1]" = var_mean_37[0]
    getitem_87: "f32[1, 1280, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_204: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_37: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_41: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_87)
    mul_302: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_37);  sub_41 = None
    squeeze_111: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_112: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_303: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_304: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_205: "f32[1280]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    squeeze_113: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_305: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0019569471624266);  squeeze_113 = None
    mul_306: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
    mul_307: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_206: "f32[1280]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    unsqueeze_148: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_149: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_308: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_149);  mul_302 = unsqueeze_149 = None
    unsqueeze_150: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_151: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_207: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_151);  mul_308 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_61: "f32[8, 1280, 8, 8]" = torch.ops.aten.clone.default(add_207)
    sigmoid_39: "f32[8, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_207)
    mul_309: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_39);  add_207 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_6: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_309, [-1, -2], True);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_96: "f32[8, 1280]" = torch.ops.aten.view.default(mean_6, [8, 1280]);  mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_62: "f32[8, 1280]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_32: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_148, clone_62, permute_32);  primals_148 = None
    permute_33: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_8: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_33);  permute_33 = None
    permute_34: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_9: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_34, clone_62);  permute_34 = clone_62 = None
    permute_35: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_5: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_97: "f32[1000]" = torch.ops.aten.view.default(sum_5, [1000]);  sum_5 = None
    permute_36: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_98: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm_8, [8, 1280, 1, 1]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_24: "f32[8, 1280, 8, 8]" = torch.ops.aten.expand.default(view_98, [8, 1280, 8, 8]);  view_98 = None
    div_4: "f32[8, 1280, 8, 8]" = torch.ops.aten.div.Scalar(expand_24, 64);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(clone_61)
    full: "f32[8, 1280, 8, 8]" = torch.ops.aten.full.default([8, 1280, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_42: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(full, sigmoid_40);  full = None
    mul_310: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(clone_61, sub_42);  clone_61 = sub_42 = None
    add_208: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Scalar(mul_310, 1);  mul_310 = None
    mul_311: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_208);  sigmoid_40 = add_208 = None
    mul_312: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(div_4, mul_311);  div_4 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_153: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    sum_6: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3])
    sub_43: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_154)
    mul_313: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(mul_312, sub_43);  sub_43 = None
    sum_7: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 2, 3]);  mul_313 = None
    mul_314: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_155: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_156: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_315: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_316: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_317: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    unsqueeze_158: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_159: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_318: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_83);  primals_83 = None
    unsqueeze_161: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_162: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    sub_44: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_154);  convolution_49 = unsqueeze_154 = None
    mul_319: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_160);  sub_44 = unsqueeze_160 = None
    sub_45: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(mul_312, mul_319);  mul_312 = mul_319 = None
    sub_46: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_157);  sub_45 = unsqueeze_157 = None
    mul_320: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_163);  sub_46 = unsqueeze_163 = None
    mul_321: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_112);  sum_7 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_320, mul_301, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = mul_301 = primals_146 = None
    getitem_88: "f32[8, 1536, 8, 8]" = convolution_backward[0]
    getitem_89: "f32[1280, 1536, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_41: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(clone_60)
    full_1: "f32[8, 1536, 8, 8]" = torch.ops.aten.full.default([8, 1536, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_47: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_41);  full_1 = None
    mul_322: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(clone_60, sub_47);  clone_60 = sub_47 = None
    add_209: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Scalar(mul_322, 1);  mul_322 = None
    mul_323: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_209);  sigmoid_41 = add_209 = None
    mul_324: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_88, mul_323);  getitem_88 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_165: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    sum_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 2, 3])
    sub_48: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_166)
    mul_325: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_324, sub_48);  sub_48 = None
    sum_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3]);  mul_325 = None
    mul_326: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_167: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_168: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_327: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_328: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_329: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
    unsqueeze_170: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
    unsqueeze_171: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_330: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_81);  primals_81 = None
    unsqueeze_173: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_174: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    sub_49: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_166);  convolution_48 = unsqueeze_166 = None
    mul_331: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_172);  sub_49 = unsqueeze_172 = None
    sub_50: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_324, mul_331);  mul_331 = None
    sub_51: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_50, unsqueeze_169);  sub_50 = unsqueeze_169 = None
    mul_332: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_175);  sub_51 = unsqueeze_175 = None
    mul_333: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_109);  sum_9 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_332, mul_293, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_332 = mul_293 = primals_145 = None
    getitem_91: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_92: "f32[1536, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_59)
    full_2: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_52: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_42);  full_2 = None
    mul_334: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_59, sub_52);  clone_59 = sub_52 = None
    add_210: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_334, 1);  mul_334 = None
    mul_335: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_210);  sigmoid_42 = add_210 = None
    mul_336: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_91, mul_335);  getitem_91 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_177: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3])
    sub_53: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_178)
    mul_337: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_336, sub_53);  sub_53 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_179: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_180: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_339: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_340: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_341: "f32[512]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_182: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_183: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_342: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_79);  primals_79 = None
    unsqueeze_185: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_186: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    sub_54: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_178);  view_95 = unsqueeze_178 = None
    mul_343: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_184);  sub_54 = unsqueeze_184 = None
    sub_55: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_336, mul_343);  mul_336 = mul_343 = None
    sub_56: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_181);  sub_55 = unsqueeze_181 = None
    mul_344: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_187);  sub_56 = unsqueeze_187 = None
    mul_345: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_106);  sum_11 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_99: "f32[32, 128, 64]" = torch.ops.aten.view.default(mul_344, [32, 128, 64]);  mul_344 = None
    permute_40: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    view_100: "f32[32, 64, 128]" = torch.ops.aten.view.default(permute_40, [32, 64, 128]);  permute_40 = None
    permute_41: "f32[32, 64, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    bmm_8: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(permute_41, view_100);  permute_41 = None
    permute_42: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    bmm_9: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(view_100, permute_42);  view_100 = permute_42 = None
    view_101: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_8, [32, 64, 128]);  bmm_8 = None
    view_102: "f32[32, 64, 64]" = torch.ops.aten.view.default(bmm_9, [32, 64, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[32, 64, 64]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_346: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(view_102, alias_16);  view_102 = None
    sum_12: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
    mul_347: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(alias_16, sum_12);  alias_16 = sum_12 = None
    sub_57: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_103: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.view.default(sub_57, [32, 8, 8, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_43: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_103, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_13: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_43, [2], True);  permute_43 = None
    view_104: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_13, [256, 8, 8]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_3: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_3, view_104, 2, 7, 9223372036854775807);  full_3 = view_104 = None
    full_4: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter, 1, 0, 8);  full_4 = slice_scatter = None
    full_5: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_1, 0, 0, 9223372036854775807);  full_5 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_105: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_16: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_105, [0, -7]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_106: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_16, [256, 8, 16]);  constant_pad_nd_16 = None
    constant_pad_nd_17: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, -1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_107: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_17, [32, 8, 8, 15]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_108: "f32[2048, 15]" = torch.ops.aten.view.default(view_107, [2048, 15]);  view_107 = None
    permute_44: "f32[15, 2048]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_10: "f32[15, 128]" = torch.ops.aten.mm.default(permute_44, view_85);  permute_44 = view_85 = None
    permute_45: "f32[128, 15]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    permute_46: "f32[15, 128]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_11: "f32[2048, 128]" = torch.ops.aten.mm.default(view_108, permute_46);  view_108 = permute_46 = None
    view_109: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_11, [32, 8, 8, 128]);  mm_11 = None
    permute_47: "f32[15, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_48: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_49: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_14: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_49, [2], True);  permute_49 = None
    view_110: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_14, [256, 8, 8]);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_6: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_3: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_6, view_110, 2, 7, 9223372036854775807);  full_6 = view_110 = None
    full_7: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_3, 1, 0, 8);  full_7 = slice_scatter_3 = None
    full_8: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_4, 0, 0, 9223372036854775807);  full_8 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_111: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_18: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_111, [0, -7]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_112: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_18, [256, 8, 16]);  constant_pad_nd_18 = None
    constant_pad_nd_19: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, -1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_113: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_19, [32, 8, 8, 15]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_114: "f32[2048, 15]" = torch.ops.aten.view.default(view_113, [2048, 15]);  view_113 = None
    permute_50: "f32[15, 2048]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_12: "f32[15, 128]" = torch.ops.aten.mm.default(permute_50, view_79);  permute_50 = view_79 = None
    permute_51: "f32[128, 15]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    permute_52: "f32[15, 128]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_13: "f32[2048, 128]" = torch.ops.aten.mm.default(view_114, permute_52);  view_114 = permute_52 = None
    view_115: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_13, [32, 8, 8, 128]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_211: "f32[32, 8, 8, 128]" = torch.ops.aten.add.Tensor(permute_48, view_115);  permute_48 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_53: "f32[15, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_63: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format);  add_211 = None
    view_116: "f32[32, 64, 128]" = torch.ops.aten.view.default(clone_63, [32, 64, 128]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_348: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(sub_57, 0.08838834764831845);  sub_57 = None
    view_117: "f32[32, 64, 64]" = torch.ops.aten.view.default(mul_348, [32, 64, 64]);  mul_348 = None
    permute_54: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_10: "f32[32, 128, 64]" = torch.ops.aten.bmm.default(permute_54, view_117);  permute_54 = None
    permute_55: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_11: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(view_117, permute_55);  view_117 = permute_55 = None
    view_118: "f32[32, 128, 64]" = torch.ops.aten.view.default(bmm_10, [32, 128, 64]);  bmm_10 = None
    view_119: "f32[32, 64, 128]" = torch.ops.aten.view.default(bmm_11, [32, 64, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_212: "f32[32, 64, 128]" = torch.ops.aten.add.Tensor(view_116, view_119);  view_116 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_56: "f32[32, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    clone_64: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_120: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_64, [8, 512, 8, 8]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_121: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(view_118, [8, 512, 8, 8]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_57: "f32[32, 128, 64]" = torch.ops.aten.permute.default(add_212, [0, 2, 1]);  add_212 = None
    clone_65: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_122: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_65, [8, 512, 8, 8]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat: "f32[8, 1536, 8, 8]" = torch.ops.aten.cat.default([view_122, view_121, view_120], 1);  view_122 = view_121 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(cat, mul_284, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = mul_284 = primals_144 = None
    getitem_94: "f32[8, 512, 8, 8]" = convolution_backward_2[0]
    getitem_95: "f32[1536, 512, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_51)
    full_9: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_43);  full_9 = None
    mul_349: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_51, sub_58);  clone_51 = sub_58 = None
    add_213: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_349, 1);  mul_349 = None
    mul_350: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_213);  sigmoid_43 = add_213 = None
    mul_351: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_94, mul_350);  getitem_94 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_189: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    sum_15: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3])
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_190)
    mul_352: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_351, sub_59);  sub_59 = None
    sum_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    unsqueeze_191: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_192: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_354: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    mul_355: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_357: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_75);  primals_75 = None
    unsqueeze_197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    sub_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_190);  convolution_46 = unsqueeze_190 = None
    mul_358: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_196);  sub_60 = unsqueeze_196 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_351, mul_358);  mul_351 = mul_358 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_193);  sub_61 = unsqueeze_193 = None
    mul_359: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_199);  sub_62 = unsqueeze_199 = None
    mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_103);  sum_16 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_359, mul_276, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_276 = primals_143 = None
    getitem_97: "f32[8, 1536, 8, 8]" = convolution_backward_3[0]
    getitem_98: "f32[512, 1536, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_214: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_324, getitem_97);  mul_324 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_44: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(clone_50)
    full_10: "f32[8, 1536, 8, 8]" = torch.ops.aten.full.default([8, 1536, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_63: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_44);  full_10 = None
    mul_361: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(clone_50, sub_63);  clone_50 = sub_63 = None
    add_215: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Scalar(mul_361, 1);  mul_361 = None
    mul_362: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_215);  sigmoid_44 = add_215 = None
    mul_363: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_214, mul_362);  add_214 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_201: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    sum_17: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3])
    sub_64: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_202)
    mul_364: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_363, sub_64);  sub_64 = None
    sum_18: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    unsqueeze_203: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_204: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_366: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    mul_367: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_368: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_206: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_207: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_369: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_73);  primals_73 = None
    unsqueeze_209: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_210: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sub_65: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_202);  convolution_45 = unsqueeze_202 = None
    mul_370: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_208);  sub_65 = unsqueeze_208 = None
    sub_66: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_363, mul_370);  mul_370 = None
    sub_67: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_205);  sub_66 = unsqueeze_205 = None
    mul_371: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_211);  sub_67 = unsqueeze_211 = None
    mul_372: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_100);  sum_18 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_371, mul_244, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = primals_142 = None
    getitem_100: "f32[8, 1024, 16, 16]" = convolution_backward_4[0]
    getitem_101: "f32[1536, 1024, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_213: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    sum_19: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3])
    sub_68: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_214)
    mul_373: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_363, sub_68);  sub_68 = None
    sum_20: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_374: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    unsqueeze_215: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_216: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_375: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    mul_376: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_377: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_218: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_219: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_378: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_71);  primals_71 = None
    unsqueeze_221: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_222: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    sub_69: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_214);  convolution_44 = unsqueeze_214 = None
    mul_379: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_220);  sub_69 = unsqueeze_220 = None
    sub_70: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(mul_363, mul_379);  mul_363 = mul_379 = None
    sub_71: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_217);  sub_70 = unsqueeze_217 = None
    mul_380: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_223);  sub_71 = unsqueeze_223 = None
    mul_381: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_97);  sum_20 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_380, mul_261, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = mul_261 = primals_141 = None
    getitem_103: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_104: "f32[1536, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_49)
    full_11: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_72: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_45);  full_11 = None
    mul_382: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_49, sub_72);  clone_49 = sub_72 = None
    add_216: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_382, 1);  mul_382 = None
    mul_383: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_216);  sigmoid_45 = add_216 = None
    mul_384: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_103, mul_383);  getitem_103 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_225: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3])
    sub_73: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_226)
    mul_385: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_384, sub_73);  sub_73 = None
    sum_22: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 2, 3]);  mul_385 = None
    mul_386: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    unsqueeze_227: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_228: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_387: "f32[512]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    mul_388: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_389: "f32[512]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_230: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_231: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_390: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_69);  primals_69 = None
    unsqueeze_233: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
    unsqueeze_234: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    sub_74: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_226);  avg_pool2d = unsqueeze_226 = None
    mul_391: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_232);  sub_74 = unsqueeze_232 = None
    sub_75: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_384, mul_391);  mul_384 = mul_391 = None
    sub_76: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_229);  sub_75 = unsqueeze_229 = None
    mul_392: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_235);  sub_76 = unsqueeze_235 = None
    mul_393: "f32[512]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_94);  sum_22 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d_backward: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(mul_392, view_71, [2, 2], [2, 2], [0, 0], False, True, None);  mul_392 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_123: "f32[32, 128, 256]" = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
    permute_61: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    view_124: "f32[32, 256, 128]" = torch.ops.aten.view.default(permute_61, [32, 256, 128]);  permute_61 = None
    permute_62: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    bmm_12: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(permute_62, view_124);  permute_62 = None
    permute_63: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    bmm_13: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_124, permute_63);  view_124 = permute_63 = None
    view_125: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_12, [32, 256, 128]);  bmm_12 = None
    view_126: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_13, [32, 256, 256]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_17: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_394: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_126, alias_17);  view_126 = None
    sum_23: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1], True)
    mul_395: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_17, sum_23);  alias_17 = sum_23 = None
    sub_77: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_127: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_77, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_64: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_24: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_64, [2], True);  permute_64 = None
    view_128: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_24, [512, 16, 16]);  sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_12: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_12, view_128, 2, 15, 9223372036854775807);  full_12 = view_128 = None
    full_13: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_13, slice_scatter_6, 1, 0, 16);  full_13 = slice_scatter_6 = None
    full_14: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_8: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_7, 0, 0, 9223372036854775807);  full_14 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_129: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_20: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_129, [0, -15]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_130: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, -1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_131: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_132: "f32[8192, 31]" = torch.ops.aten.view.default(view_131, [8192, 31]);  view_131 = None
    permute_65: "f32[31, 8192]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_14: "f32[31, 128]" = torch.ops.aten.mm.default(permute_65, view_61);  permute_65 = view_61 = None
    permute_66: "f32[128, 31]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    permute_67: "f32[31, 128]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_15: "f32[8192, 128]" = torch.ops.aten.mm.default(view_132, permute_67);  view_132 = permute_67 = None
    view_133: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_15, [32, 16, 16, 128]);  mm_15 = None
    permute_68: "f32[31, 128]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_69: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_70: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2, 4]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_25: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_70, [2], True);  permute_70 = None
    view_134: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_25, [512, 16, 16]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_15: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_9: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_15, view_134, 2, 15, 9223372036854775807);  full_15 = view_134 = None
    full_16: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_10: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_16, slice_scatter_9, 1, 0, 16);  full_16 = slice_scatter_9 = None
    full_17: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_11: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_17, slice_scatter_10, 0, 0, 9223372036854775807);  full_17 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_135: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_22: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, -15]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_136: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, -1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_137: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_138: "f32[8192, 31]" = torch.ops.aten.view.default(view_137, [8192, 31]);  view_137 = None
    permute_71: "f32[31, 8192]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_16: "f32[31, 128]" = torch.ops.aten.mm.default(permute_71, view_55);  permute_71 = view_55 = None
    permute_72: "f32[128, 31]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    permute_73: "f32[31, 128]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_17: "f32[8192, 128]" = torch.ops.aten.mm.default(view_138, permute_73);  view_138 = permute_73 = None
    view_139: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_17, [32, 16, 16, 128]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_217: "f32[32, 16, 16, 128]" = torch.ops.aten.add.Tensor(permute_69, view_139);  permute_69 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_74: "f32[31, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_66: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(add_217, memory_format = torch.contiguous_format);  add_217 = None
    view_140: "f32[32, 256, 128]" = torch.ops.aten.view.default(clone_66, [32, 256, 128]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_396: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_77, 0.08838834764831845);  sub_77 = None
    view_141: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_396, [32, 256, 256]);  mul_396 = None
    permute_75: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    bmm_14: "f32[32, 128, 256]" = torch.ops.aten.bmm.default(permute_75, view_141);  permute_75 = None
    permute_76: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_15: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(view_141, permute_76);  view_141 = permute_76 = None
    view_142: "f32[32, 128, 256]" = torch.ops.aten.view.default(bmm_14, [32, 128, 256]);  bmm_14 = None
    view_143: "f32[32, 256, 128]" = torch.ops.aten.view.default(bmm_15, [32, 256, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_218: "f32[32, 256, 128]" = torch.ops.aten.add.Tensor(view_140, view_143);  view_140 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_77: "f32[32, 128, 256]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    clone_67: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_144: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_67, [8, 512, 16, 16]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_145: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(view_142, [8, 512, 16, 16]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_78: "f32[32, 128, 256]" = torch.ops.aten.permute.default(add_218, [0, 2, 1]);  add_218 = None
    clone_68: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_146: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_68, [8, 512, 16, 16]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_1: "f32[8, 1536, 16, 16]" = torch.ops.aten.cat.default([view_146, view_145, view_144], 1);  view_146 = view_145 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(cat_1, mul_252, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = mul_252 = primals_140 = None
    getitem_106: "f32[8, 512, 16, 16]" = convolution_backward_6[0]
    getitem_107: "f32[1536, 512, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_46: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(clone_41)
    full_18: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_78: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_46);  full_18 = None
    mul_397: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(clone_41, sub_78);  clone_41 = sub_78 = None
    add_219: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_397, 1);  mul_397 = None
    mul_398: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_219);  sigmoid_46 = add_219 = None
    mul_399: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_106, mul_398);  getitem_106 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_237: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3])
    sub_79: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_238)
    mul_400: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_399, sub_79);  sub_79 = None
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    unsqueeze_239: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_240: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_402: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    mul_403: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_404: "f32[512]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_242: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_243: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_405: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_65);  primals_65 = None
    unsqueeze_245: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_246: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    sub_80: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_238);  convolution_42 = unsqueeze_238 = None
    mul_406: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_244);  sub_80 = unsqueeze_244 = None
    sub_81: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_399, mul_406);  mul_399 = mul_406 = None
    sub_82: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_241);  sub_81 = unsqueeze_241 = None
    mul_407: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_247);  sub_82 = unsqueeze_247 = None
    mul_408: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_91);  sum_27 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_407, mul_244, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = mul_244 = primals_139 = None
    getitem_109: "f32[8, 1024, 16, 16]" = convolution_backward_7[0]
    getitem_110: "f32[512, 1024, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_220: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_100, getitem_109);  getitem_100 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_47: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(clone_40)
    full_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_47);  full_19 = None
    mul_409: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(clone_40, sub_83);  clone_40 = sub_83 = None
    add_221: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_409, 1);  mul_409 = None
    mul_410: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_47, add_221);  sigmoid_47 = add_221 = None
    mul_411: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_220, mul_410);  add_220 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_249: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3])
    sub_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_250)
    mul_412: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_411, sub_84);  sub_84 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 2, 3]);  mul_412 = None
    mul_413: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    unsqueeze_251: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_252: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_414: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
    mul_415: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_416: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    unsqueeze_254: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
    unsqueeze_255: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_417: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_63);  primals_63 = None
    unsqueeze_257: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_258: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sub_85: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_250);  convolution_41 = unsqueeze_250 = None
    mul_418: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_256);  sub_85 = unsqueeze_256 = None
    sub_86: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_411, mul_418);  mul_418 = None
    sub_87: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_253);  sub_86 = unsqueeze_253 = None
    mul_419: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_259);  sub_87 = unsqueeze_259 = None
    mul_420: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_88);  sum_29 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_419, mul_236, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_419 = mul_236 = primals_138 = None
    getitem_112: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_113: "f32[1024, 256, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_39)
    full_20: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_88: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_48);  full_20 = None
    mul_421: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_39, sub_88);  clone_39 = sub_88 = None
    add_222: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_421, 1);  mul_421 = None
    mul_422: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_222);  sigmoid_48 = add_222 = None
    mul_423: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_112, mul_422);  getitem_112 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_261: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 2, 3])
    sub_89: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_262)
    mul_424: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_423, sub_89);  sub_89 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 2, 3]);  mul_424 = None
    mul_425: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
    unsqueeze_263: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_264: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
    mul_427: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    unsqueeze_266: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_267: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_269: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
    unsqueeze_270: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_90: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_262);  view_47 = unsqueeze_262 = None
    mul_430: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_268);  sub_90 = unsqueeze_268 = None
    sub_91: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_423, mul_430);  mul_423 = mul_430 = None
    sub_92: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_265);  sub_91 = unsqueeze_265 = None
    mul_431: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_271);  sub_92 = unsqueeze_271 = None
    mul_432: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_85);  sum_31 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_147: "f32[32, 64, 256]" = torch.ops.aten.view.default(mul_431, [32, 64, 256]);  mul_431 = None
    permute_82: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    view_148: "f32[32, 256, 64]" = torch.ops.aten.view.default(permute_82, [32, 256, 64]);  permute_82 = None
    permute_83: "f32[32, 256, 256]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_16: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(permute_83, view_148);  permute_83 = None
    permute_84: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_17: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(view_148, permute_84);  view_148 = permute_84 = None
    view_149: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_16, [32, 256, 64]);  bmm_16 = None
    view_150: "f32[32, 256, 256]" = torch.ops.aten.view.default(bmm_17, [32, 256, 256]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_433: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(view_150, alias_18);  view_150 = None
    sum_32: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [-1], True)
    mul_434: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_18, sum_32);  alias_18 = sum_32 = None
    sub_93: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_151: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_93, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_85: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_151, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_33: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_85, [2], True);  permute_85 = None
    view_152: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_33, [512, 16, 16]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_21: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_21, view_152, 2, 15, 9223372036854775807);  full_21 = view_152 = None
    full_22: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_22, slice_scatter_12, 1, 0, 16);  full_22 = slice_scatter_12 = None
    full_23: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_14: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_13, 0, 0, 9223372036854775807);  full_23 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_153: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_24: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_153, [0, -15]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_154: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_24, [512, 16, 32]);  constant_pad_nd_24 = None
    constant_pad_nd_25: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_154, [0, -1]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_155: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_25, [32, 16, 16, 31]);  constant_pad_nd_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_156: "f32[8192, 31]" = torch.ops.aten.view.default(view_155, [8192, 31]);  view_155 = None
    permute_86: "f32[31, 8192]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_18: "f32[31, 64]" = torch.ops.aten.mm.default(permute_86, view_37);  permute_86 = view_37 = None
    permute_87: "f32[64, 31]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    permute_88: "f32[31, 64]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_19: "f32[8192, 64]" = torch.ops.aten.mm.default(view_156, permute_88);  view_156 = permute_88 = None
    view_157: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_19, [32, 16, 16, 64]);  mm_19 = None
    permute_89: "f32[31, 64]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_90: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_91: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_151, [0, 1, 3, 2, 4]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_34: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_91, [2], True);  permute_91 = None
    view_158: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_34, [512, 16, 16]);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_24: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_15: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_24, view_158, 2, 15, 9223372036854775807);  full_24 = view_158 = None
    full_25: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_25, slice_scatter_15, 1, 0, 16);  full_25 = slice_scatter_15 = None
    full_26: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_17: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_26, slice_scatter_16, 0, 0, 9223372036854775807);  full_26 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_159: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_26: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_159, [0, -15]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_160: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_26, [512, 16, 32]);  constant_pad_nd_26 = None
    constant_pad_nd_27: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_160, [0, -1]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_161: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_27, [32, 16, 16, 31]);  constant_pad_nd_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_162: "f32[8192, 31]" = torch.ops.aten.view.default(view_161, [8192, 31]);  view_161 = None
    permute_92: "f32[31, 8192]" = torch.ops.aten.permute.default(view_162, [1, 0])
    mm_20: "f32[31, 64]" = torch.ops.aten.mm.default(permute_92, view_31);  permute_92 = view_31 = None
    permute_93: "f32[64, 31]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    permute_94: "f32[31, 64]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_21: "f32[8192, 64]" = torch.ops.aten.mm.default(view_162, permute_94);  view_162 = permute_94 = None
    view_163: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_21, [32, 16, 16, 64]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_223: "f32[32, 16, 16, 64]" = torch.ops.aten.add.Tensor(permute_90, view_163);  permute_90 = view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_95: "f32[31, 64]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_69: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(add_223, memory_format = torch.contiguous_format);  add_223 = None
    view_164: "f32[32, 256, 64]" = torch.ops.aten.view.default(clone_69, [32, 256, 64]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_435: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_93, 0.125);  sub_93 = None
    view_165: "f32[32, 256, 256]" = torch.ops.aten.view.default(mul_435, [32, 256, 256]);  mul_435 = None
    permute_96: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_18: "f32[32, 64, 256]" = torch.ops.aten.bmm.default(permute_96, view_165);  permute_96 = None
    permute_97: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_19: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(view_165, permute_97);  view_165 = permute_97 = None
    view_166: "f32[32, 64, 256]" = torch.ops.aten.view.default(bmm_18, [32, 64, 256]);  bmm_18 = None
    view_167: "f32[32, 256, 64]" = torch.ops.aten.view.default(bmm_19, [32, 256, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_224: "f32[32, 256, 64]" = torch.ops.aten.add.Tensor(view_164, view_167);  view_164 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_98: "f32[32, 64, 256]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
    clone_70: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_168: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_70, [8, 256, 16, 16]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_169: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(view_166, [8, 256, 16, 16]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_99: "f32[32, 64, 256]" = torch.ops.aten.permute.default(add_224, [0, 2, 1]);  add_224 = None
    clone_71: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    view_170: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_71, [8, 256, 16, 16]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_2: "f32[8, 768, 16, 16]" = torch.ops.aten.cat.default([view_170, view_169, view_168], 1);  view_170 = view_169 = view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(cat_2, mul_227, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = mul_227 = primals_137 = None
    getitem_115: "f32[8, 256, 16, 16]" = convolution_backward_9[0]
    getitem_116: "f32[768, 256, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_31)
    full_27: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_94: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_49);  full_27 = None
    mul_436: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_31, sub_94);  clone_31 = sub_94 = None
    add_225: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_436, 1);  mul_436 = None
    mul_437: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_225);  sigmoid_49 = add_225 = None
    mul_438: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_115, mul_437);  getitem_115 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_273: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3])
    sub_95: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_274)
    mul_439: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_438, sub_95);  sub_95 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_440: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    unsqueeze_275: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_276: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_441: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    mul_442: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_443: "f32[256]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    unsqueeze_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_444: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_57);  primals_57 = None
    unsqueeze_281: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_282: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_96: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_274);  convolution_39 = unsqueeze_274 = None
    mul_445: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_280);  sub_96 = unsqueeze_280 = None
    sub_97: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_438, mul_445);  mul_438 = mul_445 = None
    sub_98: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_277);  sub_97 = unsqueeze_277 = None
    mul_446: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_283);  sub_98 = unsqueeze_283 = None
    mul_447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_82);  sum_36 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_446, mul_219, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_446 = mul_219 = primals_136 = None
    getitem_118: "f32[8, 1024, 16, 16]" = convolution_backward_10[0]
    getitem_119: "f32[256, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_226: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_411, getitem_118);  mul_411 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(clone_30)
    full_28: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_99: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_50);  full_28 = None
    mul_448: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(clone_30, sub_99);  clone_30 = sub_99 = None
    add_227: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_448, 1);  mul_448 = None
    mul_449: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_227);  sigmoid_50 = add_227 = None
    mul_450: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_226, mul_449);  add_226 = mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_285: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 2, 3])
    sub_100: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_286)
    mul_451: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_450, sub_100);  sub_100 = None
    sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
    mul_452: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    unsqueeze_287: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_288: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_453: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    mul_454: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_455: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    unsqueeze_290: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_291: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_456: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_55);  primals_55 = None
    unsqueeze_293: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_294: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_101: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_286);  convolution_38 = unsqueeze_286 = None
    mul_457: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_292);  sub_101 = unsqueeze_292 = None
    sub_102: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_450, mul_457);  mul_457 = None
    sub_103: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_289);  sub_102 = unsqueeze_289 = None
    mul_458: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_295);  sub_103 = unsqueeze_295 = None
    mul_459: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_79);  sum_38 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_458, mul_211, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_458 = mul_211 = primals_135 = None
    getitem_121: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_122: "f32[1024, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_460: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_121, mul_210);  mul_210 = None
    mul_461: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_121, sigmoid_28);  getitem_121 = sigmoid_28 = None
    sum_39: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2, 3], True);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_19: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    sub_104: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_19)
    mul_462: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(alias_19, sub_104);  alias_19 = sub_104 = None
    mul_463: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_39, mul_462);  sum_39 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_463, relu_5, primals_133, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_463 = primals_133 = None
    getitem_124: "f32[8, 16, 1, 1]" = convolution_backward_12[0]
    getitem_125: "f32[256, 16, 1, 1]" = convolution_backward_12[1]
    getitem_126: "f32[256]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_21: "f32[8, 16, 1, 1]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_22: "f32[8, 16, 1, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    le: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(alias_22, 0);  alias_22 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le, scalar_tensor, getitem_124);  le = scalar_tensor = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where, mean_5, primals_131, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where = mean_5 = primals_131 = None
    getitem_127: "f32[8, 256, 1, 1]" = convolution_backward_13[0]
    getitem_128: "f32[16, 256, 1, 1]" = convolution_backward_13[1]
    getitem_129: "f32[16]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_25: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_127, [8, 256, 16, 16]);  getitem_127 = None
    div_5: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_25, 256);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_228: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_461, div_5);  mul_461 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_29)
    full_29: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_105: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_51);  full_29 = None
    mul_464: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_29, sub_105);  clone_29 = sub_105 = None
    add_229: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_464, 1);  mul_464 = None
    mul_465: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_229);  sigmoid_51 = add_229 = None
    mul_466: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_228, mul_465);  add_228 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_297: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 2, 3])
    sub_106: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_298)
    mul_467: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_466, sub_106);  sub_106 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    unsqueeze_299: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_300: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_469: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.00048828125)
    mul_470: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_471: "f32[256]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_302: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_303: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_472: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_53);  primals_53 = None
    unsqueeze_305: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_306: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_107: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_298);  convolution_35 = unsqueeze_298 = None
    mul_473: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_304);  sub_107 = unsqueeze_304 = None
    sub_108: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_466, mul_473);  mul_466 = mul_473 = None
    sub_109: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_301);  sub_108 = unsqueeze_301 = None
    mul_474: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_307);  sub_109 = unsqueeze_307 = None
    mul_475: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_76);  sum_41 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_474, mul_202, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = mul_202 = primals_130 = None
    getitem_130: "f32[8, 256, 16, 16]" = convolution_backward_14[0]
    getitem_131: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_28)
    full_30: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_110: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_52);  full_30 = None
    mul_476: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_28, sub_110);  clone_28 = sub_110 = None
    add_230: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_476, 1);  mul_476 = None
    mul_477: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_230);  sigmoid_52 = add_230 = None
    mul_478: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_130, mul_477);  getitem_130 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_309: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 2, 3])
    sub_111: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_310)
    mul_479: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_478, sub_111);  sub_111 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_480: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.00048828125)
    unsqueeze_311: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_312: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_481: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.00048828125)
    mul_482: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_483: "f32[256]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_314: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_315: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_484: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_317: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_318: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_112: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_310);  convolution_34 = unsqueeze_310 = None
    mul_485: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_316);  sub_112 = unsqueeze_316 = None
    sub_113: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_478, mul_485);  mul_478 = mul_485 = None
    sub_114: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_313);  sub_113 = unsqueeze_313 = None
    mul_486: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_319);  sub_114 = unsqueeze_319 = None
    mul_487: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_73);  sum_43 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_486, mul_194, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_194 = primals_129 = None
    getitem_133: "f32[8, 1024, 16, 16]" = convolution_backward_15[0]
    getitem_134: "f32[256, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_231: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_450, getitem_133);  mul_450 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_53: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(clone_27)
    full_31: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_115: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_53);  full_31 = None
    mul_488: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(clone_27, sub_115);  clone_27 = sub_115 = None
    add_232: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_488, 1);  mul_488 = None
    mul_489: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_232);  sigmoid_53 = add_232 = None
    mul_490: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_231, mul_489);  add_231 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_321: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3])
    sub_116: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_322)
    mul_491: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_490, sub_116);  sub_116 = None
    sum_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_44, 0.00048828125)
    unsqueeze_323: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_324: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_493: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, 0.00048828125)
    mul_494: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_495: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_326: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_327: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_496: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_329: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_330: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_117: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_322);  convolution_33 = unsqueeze_322 = None
    mul_497: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_328);  sub_117 = unsqueeze_328 = None
    sub_118: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_490, mul_497);  mul_497 = None
    sub_119: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_325);  sub_118 = unsqueeze_325 = None
    mul_498: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_331);  sub_119 = unsqueeze_331 = None
    mul_499: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_70);  sum_45 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_498, mul_162, primals_128, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = primals_128 = None
    getitem_136: "f32[8, 512, 32, 32]" = convolution_backward_16[0]
    getitem_137: "f32[1024, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_333: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3])
    sub_120: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_334)
    mul_500: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_490, sub_120);  sub_120 = None
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_501: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_46, 0.00048828125)
    unsqueeze_335: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_336: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_502: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
    mul_503: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_504: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_338: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_339: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_505: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_341: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_342: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_121: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_334);  convolution_32 = unsqueeze_334 = None
    mul_506: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_340);  sub_121 = unsqueeze_340 = None
    sub_122: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(mul_490, mul_506);  mul_490 = mul_506 = None
    sub_123: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_337);  sub_122 = unsqueeze_337 = None
    mul_507: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_343);  sub_123 = unsqueeze_343 = None
    mul_508: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_67);  sum_47 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_507, mul_179, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_507 = mul_179 = primals_127 = None
    getitem_139: "f32[8, 256, 16, 16]" = convolution_backward_17[0]
    getitem_140: "f32[1024, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_509: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_139, mul_178);  mul_178 = None
    mul_510: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_139, sigmoid_24);  getitem_139 = sigmoid_24 = None
    sum_48: "f32[8, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2, 3], True);  mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_23: "f32[8, 256, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_124: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_23)
    mul_511: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(alias_23, sub_124);  alias_23 = sub_124 = None
    mul_512: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_511);  sum_48 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_512, relu_4, primals_125, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_512 = primals_125 = None
    getitem_142: "f32[8, 16, 1, 1]" = convolution_backward_18[0]
    getitem_143: "f32[256, 16, 1, 1]" = convolution_backward_18[1]
    getitem_144: "f32[256]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_25: "f32[8, 16, 1, 1]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_26: "f32[8, 16, 1, 1]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    le_1: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(alias_26, 0);  alias_26 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_142);  le_1 = scalar_tensor_1 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_1, mean_4, primals_123, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = mean_4 = primals_123 = None
    getitem_145: "f32[8, 256, 1, 1]" = convolution_backward_19[0]
    getitem_146: "f32[16, 256, 1, 1]" = convolution_backward_19[1]
    getitem_147: "f32[16]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_26: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(getitem_145, [8, 256, 16, 16]);  getitem_145 = None
    div_6: "f32[8, 256, 16, 16]" = torch.ops.aten.div.Scalar(expand_26, 256);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_510, div_6);  mul_510 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_54: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(clone_26)
    full_32: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_125: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_54);  full_32 = None
    mul_513: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(clone_26, sub_125);  clone_26 = sub_125 = None
    add_234: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_513, 1);  mul_513 = None
    mul_514: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_234);  sigmoid_54 = add_234 = None
    mul_515: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_233, mul_514);  add_233 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_345: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3])
    sub_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_346)
    mul_516: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_515, sub_126);  sub_126 = None
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_517: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.00048828125)
    unsqueeze_347: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_348: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_518: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.00048828125)
    mul_519: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_350: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_351: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_45);  primals_45 = None
    unsqueeze_353: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_354: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_127: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_346);  convolution_29 = unsqueeze_346 = None
    mul_522: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_352);  sub_127 = unsqueeze_352 = None
    sub_128: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(mul_515, mul_522);  mul_515 = mul_522 = None
    sub_129: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_349);  sub_128 = unsqueeze_349 = None
    mul_523: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_355);  sub_129 = unsqueeze_355 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_64);  sum_50 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_523, mul_170, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = mul_170 = primals_122 = None
    getitem_148: "f32[8, 256, 32, 32]" = convolution_backward_20[0]
    getitem_149: "f32[256, 256, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_55: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(clone_25)
    full_33: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_130: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_55);  full_33 = None
    mul_525: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(clone_25, sub_130);  clone_25 = sub_130 = None
    add_235: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_525, 1);  mul_525 = None
    mul_526: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_235);  sigmoid_55 = add_235 = None
    mul_527: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_148, mul_526);  getitem_148 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_357: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3])
    sub_131: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_358)
    mul_528: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_527, sub_131);  sub_131 = None
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
    mul_529: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    unsqueeze_359: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
    unsqueeze_360: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_530: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
    mul_531: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_532: "f32[256]" = torch.ops.aten.mul.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
    unsqueeze_362: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_363: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_43);  primals_43 = None
    unsqueeze_365: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_366: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_132: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_358);  convolution_28 = unsqueeze_358 = None
    mul_534: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_364);  sub_132 = unsqueeze_364 = None
    sub_133: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_527, mul_534);  mul_527 = mul_534 = None
    sub_134: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_361);  sub_133 = unsqueeze_361 = None
    mul_535: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_367);  sub_134 = unsqueeze_367 = None
    mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_61);  sum_52 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_535, mul_162, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = mul_162 = primals_121 = None
    getitem_151: "f32[8, 512, 32, 32]" = convolution_backward_21[0]
    getitem_152: "f32[256, 512, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_236: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_136, getitem_151);  getitem_136 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_56: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(clone_24)
    full_34: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_135: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_56);  full_34 = None
    mul_537: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(clone_24, sub_135);  clone_24 = sub_135 = None
    add_237: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_537, 1);  mul_537 = None
    mul_538: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_237);  sigmoid_56 = add_237 = None
    mul_539: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_236, mul_538);  add_236 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_369: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3])
    sub_136: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_370)
    mul_540: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_539, sub_136);  sub_136 = None
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
    mul_541: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    unsqueeze_371: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_372: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_542: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.0001220703125)
    mul_543: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_544: "f32[512]" = torch.ops.aten.mul.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    unsqueeze_374: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_375: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_545: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_41);  primals_41 = None
    unsqueeze_377: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_378: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    sub_137: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_370);  convolution_27 = unsqueeze_370 = None
    mul_546: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_376);  sub_137 = unsqueeze_376 = None
    sub_138: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_539, mul_546);  mul_546 = None
    sub_139: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_373);  sub_138 = unsqueeze_373 = None
    mul_547: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_379);  sub_139 = unsqueeze_379 = None
    mul_548: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_58);  sum_54 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_547, mul_154, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_547 = mul_154 = primals_120 = None
    getitem_154: "f32[8, 128, 32, 32]" = convolution_backward_22[0]
    getitem_155: "f32[512, 128, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_23)
    full_35: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_140: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_57);  full_35 = None
    mul_549: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_23, sub_140);  clone_23 = sub_140 = None
    add_238: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_549, 1);  mul_549 = None
    mul_550: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_238);  sigmoid_57 = add_238 = None
    mul_551: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_154, mul_550);  getitem_154 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_381: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3])
    sub_141: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_382)
    mul_552: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_551, sub_141);  sub_141 = None
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
    mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 0.0001220703125)
    unsqueeze_383: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_384: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.0001220703125)
    mul_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_556: "f32[128]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    unsqueeze_386: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_387: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_39);  primals_39 = None
    unsqueeze_389: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_390: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    sub_142: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_382);  view_23 = unsqueeze_382 = None
    mul_558: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_388);  sub_142 = unsqueeze_388 = None
    sub_143: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_551, mul_558);  mul_551 = mul_558 = None
    sub_144: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_385);  sub_143 = unsqueeze_385 = None
    mul_559: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_391);  sub_144 = unsqueeze_391 = None
    mul_560: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_55);  sum_56 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_171: "f32[32, 32, 1024]" = torch.ops.aten.view.default(mul_559, [32, 32, 1024]);  mul_559 = None
    permute_109: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    view_172: "f32[32, 1024, 32]" = torch.ops.aten.view.default(permute_109, [32, 1024, 32]);  permute_109 = None
    permute_110: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    bmm_20: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(permute_110, view_172);  permute_110 = None
    permute_111: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    bmm_21: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_172, permute_111);  view_172 = permute_111 = None
    view_173: "f32[32, 1024, 32]" = torch.ops.aten.view.default(bmm_20, [32, 1024, 32]);  bmm_20 = None
    view_174: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(bmm_21, [32, 1024, 1024]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[32, 1024, 1024]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_561: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_174, alias_27);  view_174 = None
    sum_57: "f32[32, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [-1], True)
    mul_562: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_27, sum_57);  alias_27 = sum_57 = None
    sub_145: "f32[32, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_175: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.view.default(sub_145, [32, 32, 32, 32, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_112: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(view_175, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_58: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(permute_112, [2], True);  permute_112 = None
    view_176: "f32[1024, 32, 32]" = torch.ops.aten.view.default(sum_58, [1024, 32, 32]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_36: "f32[1024, 32, 63]" = torch.ops.aten.full.default([1024, 32, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[1024, 32, 63]" = torch.ops.aten.slice_scatter.default(full_36, view_176, 2, 31, 9223372036854775807);  full_36 = view_176 = None
    full_37: "f32[1024, 33, 63]" = torch.ops.aten.full.default([1024, 33, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_19: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_37, slice_scatter_18, 1, 0, 32);  full_37 = slice_scatter_18 = None
    full_38: "f32[1024, 33, 63]" = torch.ops.aten.full.default([1024, 33, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_20: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_38, slice_scatter_19, 0, 0, 9223372036854775807);  full_38 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_177: "f32[1024, 2079]" = torch.ops.aten.view.default(slice_scatter_20, [1024, 2079]);  slice_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_28: "f32[1024, 2048]" = torch.ops.aten.constant_pad_nd.default(view_177, [0, -31]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_178: "f32[1024, 32, 64]" = torch.ops.aten.view.default(constant_pad_nd_28, [1024, 32, 64]);  constant_pad_nd_28 = None
    constant_pad_nd_29: "f32[1024, 32, 63]" = torch.ops.aten.constant_pad_nd.default(view_178, [0, -1]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_179: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(constant_pad_nd_29, [32, 32, 32, 63]);  constant_pad_nd_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_180: "f32[32768, 63]" = torch.ops.aten.view.default(view_179, [32768, 63]);  view_179 = None
    permute_113: "f32[63, 32768]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_22: "f32[63, 32]" = torch.ops.aten.mm.default(permute_113, view_13);  permute_113 = view_13 = None
    permute_114: "f32[32, 63]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    permute_115: "f32[63, 32]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_23: "f32[32768, 32]" = torch.ops.aten.mm.default(view_180, permute_115);  view_180 = permute_115 = None
    view_181: "f32[32, 32, 32, 32]" = torch.ops.aten.view.default(mm_23, [32, 32, 32, 32]);  mm_23 = None
    permute_116: "f32[63, 32]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_117: "f32[32, 32, 32, 32]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_118: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(view_175, [0, 1, 3, 2, 4]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_59: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(permute_118, [2], True);  permute_118 = None
    view_182: "f32[1024, 32, 32]" = torch.ops.aten.view.default(sum_59, [1024, 32, 32]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_39: "f32[1024, 32, 63]" = torch.ops.aten.full.default([1024, 32, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_21: "f32[1024, 32, 63]" = torch.ops.aten.slice_scatter.default(full_39, view_182, 2, 31, 9223372036854775807);  full_39 = view_182 = None
    full_40: "f32[1024, 33, 63]" = torch.ops.aten.full.default([1024, 33, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_22: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_40, slice_scatter_21, 1, 0, 32);  full_40 = slice_scatter_21 = None
    full_41: "f32[1024, 33, 63]" = torch.ops.aten.full.default([1024, 33, 63], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_23: "f32[1024, 33, 63]" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_22, 0, 0, 9223372036854775807);  full_41 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_183: "f32[1024, 2079]" = torch.ops.aten.view.default(slice_scatter_23, [1024, 2079]);  slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_30: "f32[1024, 2048]" = torch.ops.aten.constant_pad_nd.default(view_183, [0, -31]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_184: "f32[1024, 32, 64]" = torch.ops.aten.view.default(constant_pad_nd_30, [1024, 32, 64]);  constant_pad_nd_30 = None
    constant_pad_nd_31: "f32[1024, 32, 63]" = torch.ops.aten.constant_pad_nd.default(view_184, [0, -1]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_185: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(constant_pad_nd_31, [32, 32, 32, 63]);  constant_pad_nd_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_186: "f32[32768, 63]" = torch.ops.aten.view.default(view_185, [32768, 63]);  view_185 = None
    permute_119: "f32[63, 32768]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_24: "f32[63, 32]" = torch.ops.aten.mm.default(permute_119, view_7);  permute_119 = view_7 = None
    permute_120: "f32[32, 63]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    permute_121: "f32[63, 32]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_25: "f32[32768, 32]" = torch.ops.aten.mm.default(view_186, permute_121);  view_186 = permute_121 = None
    view_187: "f32[32, 32, 32, 32]" = torch.ops.aten.view.default(mm_25, [32, 32, 32, 32]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_239: "f32[32, 32, 32, 32]" = torch.ops.aten.add.Tensor(permute_117, view_187);  permute_117 = view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_122: "f32[63, 32]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_72: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format);  add_239 = None
    view_188: "f32[32, 1024, 32]" = torch.ops.aten.view.default(clone_72, [32, 1024, 32]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_563: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_145, 0.1767766952966369);  sub_145 = None
    view_189: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(mul_563, [32, 1024, 1024]);  mul_563 = None
    permute_123: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    bmm_22: "f32[32, 32, 1024]" = torch.ops.aten.bmm.default(permute_123, view_189);  permute_123 = None
    permute_124: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    bmm_23: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(view_189, permute_124);  view_189 = permute_124 = None
    view_190: "f32[32, 32, 1024]" = torch.ops.aten.view.default(bmm_22, [32, 32, 1024]);  bmm_22 = None
    view_191: "f32[32, 1024, 32]" = torch.ops.aten.view.default(bmm_23, [32, 1024, 32]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_240: "f32[32, 1024, 32]" = torch.ops.aten.add.Tensor(view_188, view_191);  view_188 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_125: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    clone_73: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_192: "f32[8, 128, 32, 32]" = torch.ops.aten.view.default(clone_73, [8, 128, 32, 32]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_193: "f32[8, 128, 32, 32]" = torch.ops.aten.view.default(view_190, [8, 128, 32, 32]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_126: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(add_240, [0, 2, 1]);  add_240 = None
    clone_74: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    view_194: "f32[8, 128, 32, 32]" = torch.ops.aten.view.default(clone_74, [8, 128, 32, 32]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_3: "f32[8, 384, 32, 32]" = torch.ops.aten.cat.default([view_194, view_193, view_192], 1);  view_194 = view_193 = view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(cat_3, mul_145, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_3 = mul_145 = primals_119 = None
    getitem_157: "f32[8, 128, 32, 32]" = convolution_backward_23[0]
    getitem_158: "f32[384, 128, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_15)
    full_42: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_146: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_58);  full_42 = None
    mul_564: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_15, sub_146);  clone_15 = sub_146 = None
    add_241: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_564, 1);  mul_564 = None
    mul_565: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_241);  sigmoid_58 = add_241 = None
    mul_566: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_157, mul_565);  getitem_157 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_393: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_60: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3])
    sub_147: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_394)
    mul_567: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_566, sub_147);  sub_147 = None
    sum_61: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_568: "f32[128]" = torch.ops.aten.mul.Tensor(sum_60, 0.0001220703125)
    unsqueeze_395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_569: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, 0.0001220703125)
    mul_570: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_571: "f32[128]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    unsqueeze_398: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_399: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_572: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_401: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_402: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    sub_148: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_394);  convolution_25 = unsqueeze_394 = None
    mul_573: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_400);  sub_148 = unsqueeze_400 = None
    sub_149: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_566, mul_573);  mul_566 = mul_573 = None
    sub_150: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_397);  sub_149 = unsqueeze_397 = None
    mul_574: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_403);  sub_150 = unsqueeze_403 = None
    mul_575: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_52);  sum_61 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_574, mul_137, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = mul_137 = primals_118 = None
    getitem_160: "f32[8, 512, 32, 32]" = convolution_backward_24[0]
    getitem_161: "f32[128, 512, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_242: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_539, getitem_160);  mul_539 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_59: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(clone_14)
    full_43: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_151: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_59);  full_43 = None
    mul_576: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(clone_14, sub_151);  clone_14 = sub_151 = None
    add_243: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_576, 1);  mul_576 = None
    mul_577: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_59, add_243);  sigmoid_59 = add_243 = None
    mul_578: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_242, mul_577);  add_242 = mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_405: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3])
    sub_152: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_406)
    mul_579: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_578, sub_152);  sub_152 = None
    sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
    mul_580: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.0001220703125)
    unsqueeze_407: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_408: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_581: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, 0.0001220703125)
    mul_582: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_583: "f32[512]" = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    unsqueeze_410: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_411: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_584: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_413: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_414: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    sub_153: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_406);  convolution_24 = unsqueeze_406 = None
    mul_585: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_412);  sub_153 = unsqueeze_412 = None
    sub_154: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_578, mul_585);  mul_585 = None
    sub_155: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_409);  sub_154 = unsqueeze_409 = None
    mul_586: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_415);  sub_155 = unsqueeze_415 = None
    mul_587: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_49);  sum_63 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_586, mul_129, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_586 = mul_129 = primals_117 = None
    getitem_163: "f32[8, 128, 32, 32]" = convolution_backward_25[0]
    getitem_164: "f32[512, 128, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_588: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_163, mul_128);  mul_128 = None
    mul_589: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_17);  getitem_163 = sigmoid_17 = None
    sum_64: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2, 3], True);  mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_28: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_156: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_590: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_156);  alias_28 = sub_156 = None
    mul_591: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_64, mul_590);  sum_64 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_591, relu_3, primals_115, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_591 = primals_115 = None
    getitem_166: "f32[8, 8, 1, 1]" = convolution_backward_26[0]
    getitem_167: "f32[128, 8, 1, 1]" = convolution_backward_26[1]
    getitem_168: "f32[128]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_30: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_31: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_2: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_166);  le_2 = scalar_tensor_2 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_2, mean_3, primals_113, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_2 = mean_3 = primals_113 = None
    getitem_169: "f32[8, 128, 1, 1]" = convolution_backward_27[0]
    getitem_170: "f32[8, 128, 1, 1]" = convolution_backward_27[1]
    getitem_171: "f32[8]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_27: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_169, [8, 128, 32, 32]);  getitem_169 = None
    div_7: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_27, 1024);  expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_244: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_589, div_7);  mul_589 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_13)
    full_44: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_157: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_60);  full_44 = None
    mul_592: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_13, sub_157);  clone_13 = sub_157 = None
    add_245: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_592, 1);  mul_592 = None
    mul_593: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_60, add_245);  sigmoid_60 = add_245 = None
    mul_594: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_244, mul_593);  add_244 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_417: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_594, [0, 2, 3])
    sub_158: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_418)
    mul_595: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_594, sub_158);  sub_158 = None
    sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3]);  mul_595 = None
    mul_596: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, 0.0001220703125)
    unsqueeze_419: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_420: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_597: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, 0.0001220703125)
    mul_598: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_599: "f32[128]" = torch.ops.aten.mul.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_422: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_423: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_600: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_425: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_426: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_159: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_418);  convolution_21 = unsqueeze_418 = None
    mul_601: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_424);  sub_159 = unsqueeze_424 = None
    sub_160: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_594, mul_601);  mul_594 = mul_601 = None
    sub_161: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_421);  sub_160 = unsqueeze_421 = None
    mul_602: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_427);  sub_161 = unsqueeze_427 = None
    mul_603: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_46);  sum_66 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_602, mul_120, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_602 = mul_120 = primals_112 = None
    getitem_172: "f32[8, 128, 32, 32]" = convolution_backward_28[0]
    getitem_173: "f32[128, 128, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_12)
    full_45: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_162: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_45, sigmoid_61);  full_45 = None
    mul_604: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_12, sub_162);  clone_12 = sub_162 = None
    add_246: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_604, 1);  mul_604 = None
    mul_605: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_246);  sigmoid_61 = add_246 = None
    mul_606: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_172, mul_605);  getitem_172 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_429: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3])
    sub_163: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_430)
    mul_607: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_606, sub_163);  sub_163 = None
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, 0.0001220703125)
    unsqueeze_431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_609: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 0.0001220703125)
    mul_610: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_611: "f32[128]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_434: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_435: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_612: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_437: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_438: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_164: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_430);  convolution_20 = unsqueeze_430 = None
    mul_613: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_436);  sub_164 = unsqueeze_436 = None
    sub_165: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_606, mul_613);  mul_606 = mul_613 = None
    sub_166: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_433);  sub_165 = unsqueeze_433 = None
    mul_614: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_439);  sub_166 = unsqueeze_439 = None
    mul_615: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_43);  sum_68 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_614, mul_112, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = mul_112 = primals_111 = None
    getitem_175: "f32[8, 512, 32, 32]" = convolution_backward_29[0]
    getitem_176: "f32[128, 512, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_247: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_578, getitem_175);  mul_578 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_62: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(clone_11)
    full_46: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_167: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_46, sigmoid_62);  full_46 = None
    mul_616: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(clone_11, sub_167);  clone_11 = sub_167 = None
    add_248: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_616, 1);  mul_616 = None
    mul_617: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_62, add_248);  sigmoid_62 = add_248 = None
    mul_618: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_247, mul_617);  add_247 = mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_441: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_69: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3])
    sub_168: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_442)
    mul_619: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_618, sub_168);  sub_168 = None
    sum_70: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
    mul_620: "f32[512]" = torch.ops.aten.mul.Tensor(sum_69, 0.0001220703125)
    unsqueeze_443: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_444: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(sum_70, 0.0001220703125)
    mul_622: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_623: "f32[512]" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
    unsqueeze_446: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_447: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_624: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_449: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_450: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_169: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_442);  convolution_19 = unsqueeze_442 = None
    mul_625: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_448);  sub_169 = unsqueeze_448 = None
    sub_170: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_618, mul_625);  mul_625 = None
    sub_171: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_445);  sub_170 = unsqueeze_445 = None
    mul_626: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_451);  sub_171 = unsqueeze_451 = None
    mul_627: "f32[512]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_40);  sum_70 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_626, mul_80, primals_110, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_110 = None
    getitem_178: "f32[8, 256, 64, 64]" = convolution_backward_30[0]
    getitem_179: "f32[512, 256, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_453: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_71: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3])
    sub_172: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_454)
    mul_628: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_618, sub_172);  sub_172 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 2, 3]);  mul_628 = None
    mul_629: "f32[512]" = torch.ops.aten.mul.Tensor(sum_71, 0.0001220703125)
    unsqueeze_455: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_456: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_630: "f32[512]" = torch.ops.aten.mul.Tensor(sum_72, 0.0001220703125)
    mul_631: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_632: "f32[512]" = torch.ops.aten.mul.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    unsqueeze_458: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_459: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_633: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_461: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_462: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_173: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_454);  convolution_18 = unsqueeze_454 = None
    mul_634: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_460);  sub_173 = unsqueeze_460 = None
    sub_174: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(mul_618, mul_634);  mul_618 = mul_634 = None
    sub_175: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_457);  sub_174 = unsqueeze_457 = None
    mul_635: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_463);  sub_175 = unsqueeze_463 = None
    mul_636: "f32[512]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_37);  sum_72 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_635, mul_97, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_635 = mul_97 = primals_109 = None
    getitem_181: "f32[8, 128, 32, 32]" = convolution_backward_31[0]
    getitem_182: "f32[512, 128, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_637: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_181, mul_96);  mul_96 = None
    mul_638: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_181, sigmoid_13);  getitem_181 = sigmoid_13 = None
    sum_73: "f32[8, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2, 3], True);  mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_32: "f32[8, 128, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_176: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_639: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_176);  alias_32 = sub_176 = None
    mul_640: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sum_73, mul_639);  sum_73 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_640, relu_2, primals_107, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_640 = primals_107 = None
    getitem_184: "f32[8, 8, 1, 1]" = convolution_backward_32[0]
    getitem_185: "f32[128, 8, 1, 1]" = convolution_backward_32[1]
    getitem_186: "f32[128]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_34: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_35: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_3: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_184);  le_3 = scalar_tensor_3 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_3, mean_2, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_3 = mean_2 = primals_105 = None
    getitem_187: "f32[8, 128, 1, 1]" = convolution_backward_33[0]
    getitem_188: "f32[8, 128, 1, 1]" = convolution_backward_33[1]
    getitem_189: "f32[8]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_28: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(getitem_187, [8, 128, 32, 32]);  getitem_187 = None
    div_8: "f32[8, 128, 32, 32]" = torch.ops.aten.div.Scalar(expand_28, 1024);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_249: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_638, div_8);  mul_638 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_63: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(clone_10)
    full_47: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_177: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_47, sigmoid_63);  full_47 = None
    mul_641: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(clone_10, sub_177);  clone_10 = sub_177 = None
    add_250: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_641, 1);  mul_641 = None
    mul_642: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_63, add_250);  sigmoid_63 = add_250 = None
    mul_643: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_249, mul_642);  add_249 = mul_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_465: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3])
    sub_178: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466)
    mul_644: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_643, sub_178);  sub_178 = None
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
    mul_645: "f32[128]" = torch.ops.aten.mul.Tensor(sum_74, 0.0001220703125)
    unsqueeze_467: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_468: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_646: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, 0.0001220703125)
    mul_647: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_648: "f32[128]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_470: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_471: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_649: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_473: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_474: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_179: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466);  convolution_15 = unsqueeze_466 = None
    mul_650: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_472);  sub_179 = unsqueeze_472 = None
    sub_180: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(mul_643, mul_650);  mul_643 = mul_650 = None
    sub_181: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_469);  sub_180 = unsqueeze_469 = None
    mul_651: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_475);  sub_181 = unsqueeze_475 = None
    mul_652: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_34);  sum_75 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_651, mul_88, primals_104, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_651 = mul_88 = primals_104 = None
    getitem_190: "f32[8, 128, 64, 64]" = convolution_backward_34[0]
    getitem_191: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_64: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(clone_9)
    full_48: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_182: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_48, sigmoid_64);  full_48 = None
    mul_653: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(clone_9, sub_182);  clone_9 = sub_182 = None
    add_251: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_653, 1);  mul_653 = None
    mul_654: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_251);  sigmoid_64 = add_251 = None
    mul_655: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_190, mul_654);  getitem_190 = mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_477: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2, 3])
    sub_183: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478)
    mul_656: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_655, sub_183);  sub_183 = None
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 2, 3]);  mul_656 = None
    mul_657: "f32[128]" = torch.ops.aten.mul.Tensor(sum_76, 3.0517578125e-05)
    unsqueeze_479: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_480: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_658: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, 3.0517578125e-05)
    mul_659: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_660: "f32[128]" = torch.ops.aten.mul.Tensor(mul_658, mul_659);  mul_658 = mul_659 = None
    unsqueeze_482: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
    unsqueeze_483: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_661: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_184: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478);  convolution_14 = unsqueeze_478 = None
    mul_662: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_484);  sub_184 = unsqueeze_484 = None
    sub_185: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_655, mul_662);  mul_655 = mul_662 = None
    sub_186: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_481);  sub_185 = unsqueeze_481 = None
    mul_663: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_487);  sub_186 = unsqueeze_487 = None
    mul_664: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_31);  sum_77 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_663, mul_80, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_663 = mul_80 = primals_103 = None
    getitem_193: "f32[8, 256, 64, 64]" = convolution_backward_35[0]
    getitem_194: "f32[128, 256, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_252: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_178, getitem_193);  getitem_178 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_65: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_8)
    full_49: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_187: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_49, sigmoid_65);  full_49 = None
    mul_665: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_8, sub_187);  clone_8 = sub_187 = None
    add_253: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_665, 1);  mul_665 = None
    mul_666: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_253);  sigmoid_65 = add_253 = None
    mul_667: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_252, mul_666);  add_252 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_489: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3])
    sub_188: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490)
    mul_668: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_667, sub_188);  sub_188 = None
    sum_79: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3]);  mul_668 = None
    mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 3.0517578125e-05)
    unsqueeze_491: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
    unsqueeze_492: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, 3.0517578125e-05)
    mul_671: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_672: "f32[256]" = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
    unsqueeze_494: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_495: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_673: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_497: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_498: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_189: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490);  convolution_13 = unsqueeze_490 = None
    mul_674: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_496);  sub_189 = unsqueeze_496 = None
    sub_190: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_667, mul_674);  mul_674 = None
    sub_191: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_493);  sub_190 = unsqueeze_493 = None
    mul_675: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_499);  sub_191 = unsqueeze_499 = None
    mul_676: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_28);  sum_79 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_675, mul_72, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = mul_72 = primals_102 = None
    getitem_196: "f32[8, 64, 64, 64]" = convolution_backward_36[0]
    getitem_197: "f32[256, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_677: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_196, mul_71);  mul_71 = None
    mul_678: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_196, sigmoid_9);  getitem_196 = sigmoid_9 = None
    sum_80: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2, 3], True);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_36: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_192: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_679: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_192);  alias_36 = sub_192 = None
    mul_680: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sum_80, mul_679);  sum_80 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_680, relu_1, primals_100, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_680 = primals_100 = None
    getitem_199: "f32[8, 8, 1, 1]" = convolution_backward_37[0]
    getitem_200: "f32[64, 8, 1, 1]" = convolution_backward_37[1]
    getitem_201: "f32[64]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_38: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_39: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_4: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_199);  le_4 = scalar_tensor_4 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_4, mean_1, primals_98, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = mean_1 = primals_98 = None
    getitem_202: "f32[8, 64, 1, 1]" = convolution_backward_38[0]
    getitem_203: "f32[8, 64, 1, 1]" = convolution_backward_38[1]
    getitem_204: "f32[8]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_29: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_202, [8, 64, 64, 64]);  getitem_202 = None
    div_9: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_29, 4096);  expand_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_254: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_678, div_9);  mul_678 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_66: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_7)
    full_50: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_193: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_50, sigmoid_66);  full_50 = None
    mul_681: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_7, sub_193);  clone_7 = sub_193 = None
    add_255: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_681, 1);  mul_681 = None
    mul_682: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_255);  sigmoid_66 = add_255 = None
    mul_683: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_254, mul_682);  add_254 = mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_501: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_81: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3])
    sub_194: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_502)
    mul_684: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_683, sub_194);  sub_194 = None
    sum_82: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3]);  mul_684 = None
    mul_685: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, 3.0517578125e-05)
    unsqueeze_503: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_504: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_686: "f32[64]" = torch.ops.aten.mul.Tensor(sum_82, 3.0517578125e-05)
    mul_687: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_688: "f32[64]" = torch.ops.aten.mul.Tensor(mul_686, mul_687);  mul_686 = mul_687 = None
    unsqueeze_506: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_507: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_689: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_509: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_510: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_195: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_502);  convolution_10 = unsqueeze_502 = None
    mul_690: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_508);  sub_195 = unsqueeze_508 = None
    sub_196: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_683, mul_690);  mul_683 = mul_690 = None
    sub_197: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_505);  sub_196 = unsqueeze_505 = None
    mul_691: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_511);  sub_197 = unsqueeze_511 = None
    mul_692: "f32[64]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_25);  sum_82 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_691, mul_63, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = mul_63 = primals_97 = None
    getitem_205: "f32[8, 64, 64, 64]" = convolution_backward_39[0]
    getitem_206: "f32[64, 64, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_6)
    full_51: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_198: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_51, sigmoid_67);  full_51 = None
    mul_693: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_6, sub_198);  clone_6 = sub_198 = None
    add_256: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_693, 1);  mul_693 = None
    mul_694: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_256);  sigmoid_67 = add_256 = None
    mul_695: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_205, mul_694);  getitem_205 = mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_513: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_83: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3])
    sub_199: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_514)
    mul_696: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_695, sub_199);  sub_199 = None
    sum_84: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
    mul_697: "f32[64]" = torch.ops.aten.mul.Tensor(sum_83, 3.0517578125e-05)
    unsqueeze_515: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_516: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(sum_84, 3.0517578125e-05)
    mul_699: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_700: "f32[64]" = torch.ops.aten.mul.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    unsqueeze_518: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
    unsqueeze_519: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_701: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_521: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_522: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    sub_200: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_514);  convolution_9 = unsqueeze_514 = None
    mul_702: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_520);  sub_200 = unsqueeze_520 = None
    sub_201: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_695, mul_702);  mul_695 = mul_702 = None
    sub_202: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_517);  sub_201 = unsqueeze_517 = None
    mul_703: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_523);  sub_202 = unsqueeze_523 = None
    mul_704: "f32[64]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_22);  sum_84 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_703, mul_55, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = mul_55 = primals_96 = None
    getitem_208: "f32[8, 256, 64, 64]" = convolution_backward_40[0]
    getitem_209: "f32[64, 256, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_257: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_667, getitem_208);  mul_667 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_68: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_5)
    full_52: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_203: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_52, sigmoid_68);  full_52 = None
    mul_705: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_5, sub_203);  clone_5 = sub_203 = None
    add_258: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_705, 1);  mul_705 = None
    mul_706: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_258);  sigmoid_68 = add_258 = None
    mul_707: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_257, mul_706);  add_257 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_525: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3])
    sub_204: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_526)
    mul_708: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_707, sub_204);  sub_204 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 2, 3]);  mul_708 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_85, 3.0517578125e-05)
    unsqueeze_527: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_528: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, 3.0517578125e-05)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
    unsqueeze_530: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_531: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_533: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_534: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    sub_205: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_526);  convolution_8 = unsqueeze_526 = None
    mul_714: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_532);  sub_205 = unsqueeze_532 = None
    sub_206: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_707, mul_714);  mul_714 = None
    sub_207: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_529);  sub_206 = unsqueeze_529 = None
    mul_715: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_535);  sub_207 = unsqueeze_535 = None
    mul_716: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_19);  sum_86 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_715, mul_23, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_715 = primals_95 = None
    getitem_211: "f32[8, 64, 64, 64]" = convolution_backward_41[0]
    getitem_212: "f32[256, 64, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_537: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_87: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3])
    sub_208: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_538)
    mul_717: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_707, sub_208);  sub_208 = None
    sum_88: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, 3.0517578125e-05)
    unsqueeze_539: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_540: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, 3.0517578125e-05)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_542: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_543: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_545: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_546: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    sub_209: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_538);  convolution_7 = unsqueeze_538 = None
    mul_723: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_544);  sub_209 = unsqueeze_544 = None
    sub_210: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_707, mul_723);  mul_707 = mul_723 = None
    sub_211: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_541);  sub_210 = unsqueeze_541 = None
    mul_724: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_547);  sub_211 = unsqueeze_547 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_16);  sum_88 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_724, mul_40, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = mul_40 = primals_94 = None
    getitem_214: "f32[8, 64, 64, 64]" = convolution_backward_42[0]
    getitem_215: "f32[256, 64, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_726: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_214, mul_39);  mul_39 = None
    mul_727: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_214, sigmoid_5);  getitem_214 = sigmoid_5 = None
    sum_89: "f32[8, 64, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [2, 3], True);  mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_40: "f32[8, 64, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_212: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_40)
    mul_728: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(alias_40, sub_212);  alias_40 = sub_212 = None
    mul_729: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sum_89, mul_728);  sum_89 = mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_729, relu, primals_92, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_729 = primals_92 = None
    getitem_217: "f32[8, 8, 1, 1]" = convolution_backward_43[0]
    getitem_218: "f32[64, 8, 1, 1]" = convolution_backward_43[1]
    getitem_219: "f32[64]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_42: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_43: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_5: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_217);  le_5 = scalar_tensor_5 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_5, mean, primals_90, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = mean = primals_90 = None
    getitem_220: "f32[8, 64, 1, 1]" = convolution_backward_44[0]
    getitem_221: "f32[8, 64, 1, 1]" = convolution_backward_44[1]
    getitem_222: "f32[8]" = convolution_backward_44[2];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_30: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(getitem_220, [8, 64, 64, 64]);  getitem_220 = None
    div_10: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Scalar(expand_30, 4096);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_259: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_727, div_10);  mul_727 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_69: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_4)
    full_53: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_213: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_53, sigmoid_69);  full_53 = None
    mul_730: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_4, sub_213);  clone_4 = sub_213 = None
    add_260: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_730, 1);  mul_730 = None
    mul_731: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_260);  sigmoid_69 = add_260 = None
    mul_732: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_259, mul_731);  add_259 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_549: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_90: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3])
    sub_214: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_550)
    mul_733: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_732, sub_214);  sub_214 = None
    sum_91: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_734: "f32[64]" = torch.ops.aten.mul.Tensor(sum_90, 3.0517578125e-05)
    unsqueeze_551: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_552: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_735: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, 3.0517578125e-05)
    mul_736: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_737: "f32[64]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_554: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_555: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_738: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_557: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_558: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    sub_215: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_550);  convolution_4 = unsqueeze_550 = None
    mul_739: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_556);  sub_215 = unsqueeze_556 = None
    sub_216: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_732, mul_739);  mul_732 = mul_739 = None
    sub_217: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_553);  sub_216 = unsqueeze_553 = None
    mul_740: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_559);  sub_217 = unsqueeze_559 = None
    mul_741: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_13);  sum_91 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_740, mul_31, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_740 = mul_31 = primals_89 = None
    getitem_223: "f32[8, 64, 64, 64]" = convolution_backward_45[0]
    getitem_224: "f32[64, 64, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_70: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_3)
    full_54: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_218: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_54, sigmoid_70);  full_54 = None
    mul_742: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_3, sub_218);  clone_3 = sub_218 = None
    add_261: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_742, 1);  mul_742 = None
    mul_743: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_261);  sigmoid_70 = add_261 = None
    mul_744: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_223, mul_743);  getitem_223 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_561: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_92: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3])
    sub_219: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_562)
    mul_745: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_744, sub_219);  sub_219 = None
    sum_93: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3]);  mul_745 = None
    mul_746: "f32[64]" = torch.ops.aten.mul.Tensor(sum_92, 3.0517578125e-05)
    unsqueeze_563: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_564: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_747: "f32[64]" = torch.ops.aten.mul.Tensor(sum_93, 3.0517578125e-05)
    mul_748: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_749: "f32[64]" = torch.ops.aten.mul.Tensor(mul_747, mul_748);  mul_747 = mul_748 = None
    unsqueeze_566: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_567: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_750: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_569: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_570: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sub_220: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_562);  convolution_3 = unsqueeze_562 = None
    mul_751: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_568);  sub_220 = unsqueeze_568 = None
    sub_221: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_744, mul_751);  mul_744 = mul_751 = None
    sub_222: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_565);  sub_221 = unsqueeze_565 = None
    mul_752: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_571);  sub_222 = unsqueeze_571 = None
    mul_753: "f32[64]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_10);  sum_93 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_752, mul_23, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = mul_23 = primals_88 = None
    getitem_226: "f32[8, 64, 64, 64]" = convolution_backward_46[0]
    getitem_227: "f32[64, 64, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_262: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_211, getitem_226);  getitem_211 = getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_71: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(clone_2)
    full_55: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_223: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_55, sigmoid_71);  full_55 = None
    mul_754: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(clone_2, sub_223);  clone_2 = sub_223 = None
    add_263: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_754, 1);  mul_754 = None
    mul_755: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_263);  sigmoid_71 = add_263 = None
    mul_756: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_262, mul_755);  add_262 = mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_573: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_94: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_756, [0, 2, 3])
    sub_224: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_574)
    mul_757: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_756, sub_224);  sub_224 = None
    sum_95: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[64]" = torch.ops.aten.mul.Tensor(sum_94, 3.0517578125e-05)
    unsqueeze_575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_759: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, 3.0517578125e-05)
    mul_760: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_761: "f32[64]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    sub_225: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_574);  convolution_2 = unsqueeze_574 = None
    mul_763: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_580);  sub_225 = unsqueeze_580 = None
    sub_226: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(mul_756, mul_763);  mul_756 = mul_763 = None
    sub_227: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_577);  sub_226 = unsqueeze_577 = None
    mul_764: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_583);  sub_227 = unsqueeze_583 = None
    mul_765: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_7);  sum_95 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_764, mul_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = mul_15 = primals_87 = None
    getitem_229: "f32[8, 32, 128, 128]" = convolution_backward_47[0]
    getitem_230: "f32[64, 32, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_72: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(clone_1)
    full_56: "f32[8, 32, 128, 128]" = torch.ops.aten.full.default([8, 32, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_228: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(full_56, sigmoid_72);  full_56 = None
    mul_766: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(clone_1, sub_228);  clone_1 = sub_228 = None
    add_264: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Scalar(mul_766, 1);  mul_766 = None
    mul_767: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_264);  sigmoid_72 = add_264 = None
    mul_768: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_229, mul_767);  getitem_229 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_585: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_96: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 2, 3])
    sub_229: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_586)
    mul_769: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_768, sub_229);  sub_229 = None
    sum_97: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_770: "f32[32]" = torch.ops.aten.mul.Tensor(sum_96, 7.62939453125e-06)
    unsqueeze_587: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_588: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_771: "f32[32]" = torch.ops.aten.mul.Tensor(sum_97, 7.62939453125e-06)
    mul_772: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_773: "f32[32]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_590: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_591: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_774: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_593: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_594: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    sub_230: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_586);  convolution_1 = unsqueeze_586 = None
    mul_775: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_592);  sub_230 = unsqueeze_592 = None
    sub_231: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(mul_768, mul_775);  mul_768 = mul_775 = None
    sub_232: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_589);  sub_231 = unsqueeze_589 = None
    mul_776: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_595);  sub_232 = unsqueeze_595 = None
    mul_777: "f32[32]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_4);  sum_97 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_776, mul_7, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = mul_7 = primals_86 = None
    getitem_232: "f32[8, 24, 128, 128]" = convolution_backward_48[0]
    getitem_233: "f32[32, 24, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(clone)
    full_57: "f32[8, 24, 128, 128]" = torch.ops.aten.full.default([8, 24, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_233: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(full_57, sigmoid_73);  full_57 = None
    mul_778: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(clone, sub_233);  clone = sub_233 = None
    add_265: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Scalar(mul_778, 1);  mul_778 = None
    mul_779: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_265);  sigmoid_73 = add_265 = None
    mul_780: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_232, mul_779);  getitem_232 = mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_597: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_98: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3])
    sub_234: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_598)
    mul_781: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_780, sub_234);  sub_234 = None
    sum_99: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2, 3]);  mul_781 = None
    mul_782: "f32[24]" = torch.ops.aten.mul.Tensor(sum_98, 7.62939453125e-06)
    unsqueeze_599: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_600: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_783: "f32[24]" = torch.ops.aten.mul.Tensor(sum_99, 7.62939453125e-06)
    mul_784: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_785: "f32[24]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    unsqueeze_602: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_603: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_786: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_605: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_606: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    sub_235: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_598);  convolution = unsqueeze_598 = None
    mul_787: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_604);  sub_235 = unsqueeze_604 = None
    sub_236: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(mul_780, mul_787);  mul_780 = mul_787 = None
    sub_237: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_601);  sub_236 = unsqueeze_601 = None
    mul_788: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_607);  sub_237 = unsqueeze_607 = None
    mul_789: "f32[24]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_1);  sum_99 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_788, primals_263, primals_85, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_788 = primals_263 = primals_85 = None
    getitem_236: "f32[24, 3, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_149, add);  primals_149 = add = None
    copy__1: "f32[24]" = torch.ops.aten.copy_.default(primals_150, add_2);  primals_150 = add_2 = None
    copy__2: "f32[24]" = torch.ops.aten.copy_.default(primals_151, add_3);  primals_151 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_152, add_5);  primals_152 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_153, add_7);  primals_153 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_154, add_8);  primals_154 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_155, add_10);  primals_155 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_156, add_12);  primals_156 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_157, add_13);  primals_157 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_15);  primals_158 = add_15 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_159, add_17);  primals_159 = add_17 = None
    copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_160, add_18);  primals_160 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_161, add_20);  primals_161 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_162, add_22);  primals_162 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_163, add_23);  primals_163 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_164, add_25);  primals_164 = add_25 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_165, add_27);  primals_165 = add_27 = None
    copy__17: "f32[256]" = torch.ops.aten.copy_.default(primals_166, add_28);  primals_166 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_30);  primals_167 = add_30 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_168, add_32);  primals_168 = add_32 = None
    copy__20: "f32[256]" = torch.ops.aten.copy_.default(primals_169, add_33);  primals_169 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_170, add_36);  primals_170 = add_36 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_171, add_38);  primals_171 = add_38 = None
    copy__23: "f32[64]" = torch.ops.aten.copy_.default(primals_172, add_39);  primals_172 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_41);  primals_173 = add_41 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_174, add_43);  primals_174 = add_43 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_175, add_44);  primals_175 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_176, add_46);  primals_176 = add_46 = None
    copy__28: "f32[256]" = torch.ops.aten.copy_.default(primals_177, add_48);  primals_177 = add_48 = None
    copy__29: "f32[256]" = torch.ops.aten.copy_.default(primals_178, add_49);  primals_178 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_52);  primals_179 = add_52 = None
    copy__31: "f32[128]" = torch.ops.aten.copy_.default(primals_180, add_54);  primals_180 = add_54 = None
    copy__32: "f32[128]" = torch.ops.aten.copy_.default(primals_181, add_55);  primals_181 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_182, add_57);  primals_182 = add_57 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_183, add_59);  primals_183 = add_59 = None
    copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_184, add_60);  primals_184 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_62);  primals_185 = add_62 = None
    copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_186, add_64);  primals_186 = add_64 = None
    copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_187, add_65);  primals_187 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_188, add_67);  primals_188 = add_67 = None
    copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_189, add_69);  primals_189 = add_69 = None
    copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_190, add_70);  primals_190 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_73);  primals_191 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_192, add_75);  primals_192 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_193, add_76);  primals_193 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_194, add_78);  primals_194 = add_78 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_195, add_80);  primals_195 = add_80 = None
    copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_196, add_81);  primals_196 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_83);  primals_197 = add_83 = None
    copy__49: "f32[512]" = torch.ops.aten.copy_.default(primals_198, add_85);  primals_198 = add_85 = None
    copy__50: "f32[512]" = torch.ops.aten.copy_.default(primals_199, add_86);  primals_199 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_200, add_89);  primals_200 = add_89 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_201, add_91);  primals_201 = add_91 = None
    copy__53: "f32[128]" = torch.ops.aten.copy_.default(primals_202, add_92);  primals_202 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_96);  primals_203 = add_96 = None
    copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_204, add_98);  primals_204 = add_98 = None
    copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_205, add_99);  primals_205 = add_99 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_206, add_101);  primals_206 = add_101 = None
    copy__58: "f32[512]" = torch.ops.aten.copy_.default(primals_207, add_103);  primals_207 = add_103 = None
    copy__59: "f32[512]" = torch.ops.aten.copy_.default(primals_208, add_104);  primals_208 = add_104 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_107);  primals_209 = add_107 = None
    copy__61: "f32[256]" = torch.ops.aten.copy_.default(primals_210, add_109);  primals_210 = add_109 = None
    copy__62: "f32[256]" = torch.ops.aten.copy_.default(primals_211, add_110);  primals_211 = add_110 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_212, add_112);  primals_212 = add_112 = None
    copy__64: "f32[256]" = torch.ops.aten.copy_.default(primals_213, add_114);  primals_213 = add_114 = None
    copy__65: "f32[256]" = torch.ops.aten.copy_.default(primals_214, add_115);  primals_214 = add_115 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_117);  primals_215 = add_117 = None
    copy__67: "f32[1024]" = torch.ops.aten.copy_.default(primals_216, add_119);  primals_216 = add_119 = None
    copy__68: "f32[1024]" = torch.ops.aten.copy_.default(primals_217, add_120);  primals_217 = add_120 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_218, add_122);  primals_218 = add_122 = None
    copy__70: "f32[1024]" = torch.ops.aten.copy_.default(primals_219, add_124);  primals_219 = add_124 = None
    copy__71: "f32[1024]" = torch.ops.aten.copy_.default(primals_220, add_125);  primals_220 = add_125 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_221, add_128);  primals_221 = add_128 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_222, add_130);  primals_222 = add_130 = None
    copy__74: "f32[256]" = torch.ops.aten.copy_.default(primals_223, add_131);  primals_223 = add_131 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_224, add_133);  primals_224 = add_133 = None
    copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_225, add_135);  primals_225 = add_135 = None
    copy__77: "f32[256]" = torch.ops.aten.copy_.default(primals_226, add_136);  primals_226 = add_136 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_227, add_138);  primals_227 = add_138 = None
    copy__79: "f32[1024]" = torch.ops.aten.copy_.default(primals_228, add_140);  primals_228 = add_140 = None
    copy__80: "f32[1024]" = torch.ops.aten.copy_.default(primals_229, add_141);  primals_229 = add_141 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_230, add_144);  primals_230 = add_144 = None
    copy__82: "f32[256]" = torch.ops.aten.copy_.default(primals_231, add_146);  primals_231 = add_146 = None
    copy__83: "f32[256]" = torch.ops.aten.copy_.default(primals_232, add_147);  primals_232 = add_147 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_233, add_151);  primals_233 = add_151 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_234, add_153);  primals_234 = add_153 = None
    copy__86: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_154);  primals_235 = add_154 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_236, add_156);  primals_236 = add_156 = None
    copy__88: "f32[1024]" = torch.ops.aten.copy_.default(primals_237, add_158);  primals_237 = add_158 = None
    copy__89: "f32[1024]" = torch.ops.aten.copy_.default(primals_238, add_159);  primals_238 = add_159 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_162);  primals_239 = add_162 = None
    copy__91: "f32[512]" = torch.ops.aten.copy_.default(primals_240, add_164);  primals_240 = add_164 = None
    copy__92: "f32[512]" = torch.ops.aten.copy_.default(primals_241, add_165);  primals_241 = add_165 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_242, add_169);  primals_242 = add_169 = None
    copy__94: "f32[512]" = torch.ops.aten.copy_.default(primals_243, add_171);  primals_243 = add_171 = None
    copy__95: "f32[512]" = torch.ops.aten.copy_.default(primals_244, add_172);  primals_244 = add_172 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_174);  primals_245 = add_174 = None
    copy__97: "f32[1536]" = torch.ops.aten.copy_.default(primals_246, add_176);  primals_246 = add_176 = None
    copy__98: "f32[1536]" = torch.ops.aten.copy_.default(primals_247, add_177);  primals_247 = add_177 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_248, add_179);  primals_248 = add_179 = None
    copy__100: "f32[1536]" = torch.ops.aten.copy_.default(primals_249, add_181);  primals_249 = add_181 = None
    copy__101: "f32[1536]" = torch.ops.aten.copy_.default(primals_250, add_182);  primals_250 = add_182 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_251, add_185);  primals_251 = add_185 = None
    copy__103: "f32[512]" = torch.ops.aten.copy_.default(primals_252, add_187);  primals_252 = add_187 = None
    copy__104: "f32[512]" = torch.ops.aten.copy_.default(primals_253, add_188);  primals_253 = add_188 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_254, add_192);  primals_254 = add_192 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_255, add_194);  primals_255 = add_194 = None
    copy__107: "f32[512]" = torch.ops.aten.copy_.default(primals_256, add_195);  primals_256 = add_195 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_197);  primals_257 = add_197 = None
    copy__109: "f32[1536]" = torch.ops.aten.copy_.default(primals_258, add_199);  primals_258 = add_199 = None
    copy__110: "f32[1536]" = torch.ops.aten.copy_.default(primals_259, add_200);  primals_259 = add_200 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_203);  primals_260 = add_203 = None
    copy__112: "f32[1280]" = torch.ops.aten.copy_.default(primals_261, add_205);  primals_261 = add_205 = None
    copy__113: "f32[1280]" = torch.ops.aten.copy_.default(primals_262, add_206);  primals_262 = add_206 = None
    return pytree.tree_unflatten([addmm, mul_789, sum_98, mul_777, sum_96, mul_765, sum_94, mul_753, sum_92, mul_741, sum_90, mul_725, sum_87, mul_716, sum_85, mul_704, sum_83, mul_692, sum_81, mul_676, sum_78, mul_664, sum_76, mul_652, sum_74, mul_636, sum_71, mul_627, sum_69, mul_615, sum_67, mul_603, sum_65, mul_587, sum_62, mul_575, sum_60, permute_122, permute_116, mul_560, sum_55, mul_548, sum_53, mul_536, sum_51, mul_524, sum_49, mul_508, sum_46, mul_499, sum_44, mul_487, sum_42, mul_475, sum_40, mul_459, sum_37, mul_447, sum_35, permute_95, permute_89, mul_432, sum_30, mul_420, sum_28, mul_408, sum_26, permute_74, permute_68, mul_393, sum_21, mul_381, sum_19, mul_372, sum_17, mul_360, sum_15, permute_53, permute_47, mul_345, sum_10, mul_333, sum_8, mul_321, sum_6, getitem_236, getitem_233, getitem_230, getitem_227, getitem_224, getitem_221, getitem_222, getitem_218, getitem_219, getitem_215, getitem_212, getitem_209, getitem_206, getitem_203, getitem_204, getitem_200, getitem_201, getitem_197, getitem_194, getitem_191, getitem_188, getitem_189, getitem_185, getitem_186, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_171, getitem_167, getitem_168, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_147, getitem_143, getitem_144, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_129, getitem_125, getitem_126, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, permute_36, view_97, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    