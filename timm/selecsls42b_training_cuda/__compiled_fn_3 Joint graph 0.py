from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32, 3, 3, 3]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[64, 32, 3, 3]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64, 64, 1, 1]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[32, 64, 3, 3]"; primals_11: "f32[32]"; primals_12: "f32[32]"; primals_13: "f32[64, 32, 1, 1]"; primals_14: "f32[64]"; primals_15: "f32[64]"; primals_16: "f32[32, 64, 3, 3]"; primals_17: "f32[32]"; primals_18: "f32[32]"; primals_19: "f32[64, 128, 1, 1]"; primals_20: "f32[64]"; primals_21: "f32[64]"; primals_22: "f32[64, 64, 3, 3]"; primals_23: "f32[64]"; primals_24: "f32[64]"; primals_25: "f32[64, 64, 1, 1]"; primals_26: "f32[64]"; primals_27: "f32[64]"; primals_28: "f32[32, 64, 3, 3]"; primals_29: "f32[32]"; primals_30: "f32[32]"; primals_31: "f32[64, 32, 1, 1]"; primals_32: "f32[64]"; primals_33: "f32[64]"; primals_34: "f32[32, 64, 3, 3]"; primals_35: "f32[32]"; primals_36: "f32[32]"; primals_37: "f32[128, 192, 1, 1]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[144, 128, 3, 3]"; primals_41: "f32[144]"; primals_42: "f32[144]"; primals_43: "f32[144, 144, 1, 1]"; primals_44: "f32[144]"; primals_45: "f32[144]"; primals_46: "f32[72, 144, 3, 3]"; primals_47: "f32[72]"; primals_48: "f32[72]"; primals_49: "f32[144, 72, 1, 1]"; primals_50: "f32[144]"; primals_51: "f32[144]"; primals_52: "f32[72, 144, 3, 3]"; primals_53: "f32[72]"; primals_54: "f32[72]"; primals_55: "f32[144, 288, 1, 1]"; primals_56: "f32[144]"; primals_57: "f32[144]"; primals_58: "f32[144, 144, 3, 3]"; primals_59: "f32[144]"; primals_60: "f32[144]"; primals_61: "f32[144, 144, 1, 1]"; primals_62: "f32[144]"; primals_63: "f32[144]"; primals_64: "f32[72, 144, 3, 3]"; primals_65: "f32[72]"; primals_66: "f32[72]"; primals_67: "f32[144, 72, 1, 1]"; primals_68: "f32[144]"; primals_69: "f32[144]"; primals_70: "f32[72, 144, 3, 3]"; primals_71: "f32[72]"; primals_72: "f32[72]"; primals_73: "f32[288, 432, 1, 1]"; primals_74: "f32[288]"; primals_75: "f32[288]"; primals_76: "f32[304, 288, 3, 3]"; primals_77: "f32[304]"; primals_78: "f32[304]"; primals_79: "f32[304, 304, 1, 1]"; primals_80: "f32[304]"; primals_81: "f32[304]"; primals_82: "f32[152, 304, 3, 3]"; primals_83: "f32[152]"; primals_84: "f32[152]"; primals_85: "f32[304, 152, 1, 1]"; primals_86: "f32[304]"; primals_87: "f32[304]"; primals_88: "f32[152, 304, 3, 3]"; primals_89: "f32[152]"; primals_90: "f32[152]"; primals_91: "f32[304, 608, 1, 1]"; primals_92: "f32[304]"; primals_93: "f32[304]"; primals_94: "f32[304, 304, 3, 3]"; primals_95: "f32[304]"; primals_96: "f32[304]"; primals_97: "f32[304, 304, 1, 1]"; primals_98: "f32[304]"; primals_99: "f32[304]"; primals_100: "f32[152, 304, 3, 3]"; primals_101: "f32[152]"; primals_102: "f32[152]"; primals_103: "f32[304, 152, 1, 1]"; primals_104: "f32[304]"; primals_105: "f32[304]"; primals_106: "f32[152, 304, 3, 3]"; primals_107: "f32[152]"; primals_108: "f32[152]"; primals_109: "f32[480, 912, 1, 1]"; primals_110: "f32[480]"; primals_111: "f32[480]"; primals_112: "f32[960, 480, 3, 3]"; primals_113: "f32[960]"; primals_114: "f32[960]"; primals_115: "f32[1024, 960, 3, 3]"; primals_116: "f32[1024]"; primals_117: "f32[1024]"; primals_118: "f32[1280, 1024, 3, 3]"; primals_119: "f32[1280]"; primals_120: "f32[1280]"; primals_121: "f32[1024, 1280, 1, 1]"; primals_122: "f32[1024]"; primals_123: "f32[1024]"; primals_124: "f32[1000, 1024]"; primals_125: "f32[1000]"; primals_126: "f32[32]"; primals_127: "f32[32]"; primals_128: "i64[]"; primals_129: "f32[64]"; primals_130: "f32[64]"; primals_131: "i64[]"; primals_132: "f32[64]"; primals_133: "f32[64]"; primals_134: "i64[]"; primals_135: "f32[32]"; primals_136: "f32[32]"; primals_137: "i64[]"; primals_138: "f32[64]"; primals_139: "f32[64]"; primals_140: "i64[]"; primals_141: "f32[32]"; primals_142: "f32[32]"; primals_143: "i64[]"; primals_144: "f32[64]"; primals_145: "f32[64]"; primals_146: "i64[]"; primals_147: "f32[64]"; primals_148: "f32[64]"; primals_149: "i64[]"; primals_150: "f32[64]"; primals_151: "f32[64]"; primals_152: "i64[]"; primals_153: "f32[32]"; primals_154: "f32[32]"; primals_155: "i64[]"; primals_156: "f32[64]"; primals_157: "f32[64]"; primals_158: "i64[]"; primals_159: "f32[32]"; primals_160: "f32[32]"; primals_161: "i64[]"; primals_162: "f32[128]"; primals_163: "f32[128]"; primals_164: "i64[]"; primals_165: "f32[144]"; primals_166: "f32[144]"; primals_167: "i64[]"; primals_168: "f32[144]"; primals_169: "f32[144]"; primals_170: "i64[]"; primals_171: "f32[72]"; primals_172: "f32[72]"; primals_173: "i64[]"; primals_174: "f32[144]"; primals_175: "f32[144]"; primals_176: "i64[]"; primals_177: "f32[72]"; primals_178: "f32[72]"; primals_179: "i64[]"; primals_180: "f32[144]"; primals_181: "f32[144]"; primals_182: "i64[]"; primals_183: "f32[144]"; primals_184: "f32[144]"; primals_185: "i64[]"; primals_186: "f32[144]"; primals_187: "f32[144]"; primals_188: "i64[]"; primals_189: "f32[72]"; primals_190: "f32[72]"; primals_191: "i64[]"; primals_192: "f32[144]"; primals_193: "f32[144]"; primals_194: "i64[]"; primals_195: "f32[72]"; primals_196: "f32[72]"; primals_197: "i64[]"; primals_198: "f32[288]"; primals_199: "f32[288]"; primals_200: "i64[]"; primals_201: "f32[304]"; primals_202: "f32[304]"; primals_203: "i64[]"; primals_204: "f32[304]"; primals_205: "f32[304]"; primals_206: "i64[]"; primals_207: "f32[152]"; primals_208: "f32[152]"; primals_209: "i64[]"; primals_210: "f32[304]"; primals_211: "f32[304]"; primals_212: "i64[]"; primals_213: "f32[152]"; primals_214: "f32[152]"; primals_215: "i64[]"; primals_216: "f32[304]"; primals_217: "f32[304]"; primals_218: "i64[]"; primals_219: "f32[304]"; primals_220: "f32[304]"; primals_221: "i64[]"; primals_222: "f32[304]"; primals_223: "f32[304]"; primals_224: "i64[]"; primals_225: "f32[152]"; primals_226: "f32[152]"; primals_227: "i64[]"; primals_228: "f32[304]"; primals_229: "f32[304]"; primals_230: "i64[]"; primals_231: "f32[152]"; primals_232: "f32[152]"; primals_233: "i64[]"; primals_234: "f32[480]"; primals_235: "f32[480]"; primals_236: "i64[]"; primals_237: "f32[960]"; primals_238: "f32[960]"; primals_239: "i64[]"; primals_240: "f32[1024]"; primals_241: "f32[1024]"; primals_242: "i64[]"; primals_243: "f32[1280]"; primals_244: "f32[1280]"; primals_245: "i64[]"; primals_246: "f32[1024]"; primals_247: "f32[1024]"; primals_248: "i64[]"; primals_249: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:169, code: x = self.stem(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_249, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_128, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_1: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_131, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_2: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_134, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    relu_2: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    convolution_3: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_137, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_21: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[32]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    relu_3: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_4: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_140, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_28: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[64]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    relu_4: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    convolution_5: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_143, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 32, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_35: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[32]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
    add_27: "f32[32]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[32]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[32]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
    add_28: "f32[32]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_21: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_23: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    relu_5: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    cat: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([relu_1, relu_3, relu_5], 1)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_146, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_42: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
    add_32: "f32[64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[64]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    relu_6: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_149, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_49: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
    add_37: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[64]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    relu_7: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_8: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_152, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_42: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    relu_8: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    convolution_9: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_155, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 32, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_46: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_9: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_63: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[32]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_47: "f32[32]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[32]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[32]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
    add_48: "f32[32]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_37: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_39: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_49: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    relu_9: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_10: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_50: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 64, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 64, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_51: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_10: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_70: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[64]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_52: "f32[64]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
    add_53: "f32[64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_54: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    relu_10: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    convolution_11: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_10, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_161, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 32, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_56: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_11: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_77: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[32]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_57: "f32[32]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[32]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[32]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
    add_58: "f32[32]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_59: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    relu_11: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    cat_1: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_9, relu_11, relu_6], 1)
    convolution_12: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(cat_1, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_12: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_84: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
    add_63: "f32[128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_64: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    relu_12: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_64);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_13: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_40, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_65: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 144, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 144, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_66: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_13: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_91: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[144]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_67: "f32[144]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_95: "f32[144]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[144]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_68: "f32[144]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_53: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_55: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_69: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    relu_13: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_14: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_70: "i64[]" = torch.ops.aten.add.Tensor(primals_170, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 144, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 144, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_71: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_14: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_98: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[144]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_72: "f32[144]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_102: "f32[144]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[144]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
    add_73: "f32[144]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_57: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_59: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_74: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    relu_14: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    convolution_15: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 72, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 72, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_76: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_15: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_105: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[72]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_77: "f32[72]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_109: "f32[72]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[72]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
    add_78: "f32[72]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_61: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_63: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_79: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    relu_15: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_16: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_80: "i64[]" = torch.ops.aten.add.Tensor(primals_176, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 144, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 144, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_16: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_112: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[144]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_82: "f32[144]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_116: "f32[144]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[144]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_83: "f32[144]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_65: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_67: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_84: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    relu_16: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    convolution_17: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 72, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 72, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_86: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_17: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_119: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[72]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_87: "f32[72]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_123: "f32[72]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[72]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_88: "f32[72]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_69: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_71: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_89: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    relu_17: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_89);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    cat_2: "f32[8, 288, 28, 28]" = torch.ops.aten.cat.default([relu_13, relu_15, relu_17], 1)
    convolution_18: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(cat_2, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_182, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 144, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 144, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_91: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_18: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_126: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[144]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_92: "f32[144]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[144]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[144]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_93: "f32[144]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_73: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_75: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_94: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    relu_18: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_19: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 144, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 144, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_96: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_19: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
    mul_133: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[144]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_97: "f32[144]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_136: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[144]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[144]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_98: "f32[144]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_77: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_79: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_99: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    relu_19: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_20: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_188, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 144, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 144, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_101: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_20: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
    mul_140: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[144]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_102: "f32[144]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_143: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[144]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[144]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_103: "f32[144]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_81: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_83: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_104: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    relu_20: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    convolution_21: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_64, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 72, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 72, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_106: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_21: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
    mul_147: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[72]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_107: "f32[72]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_150: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[72]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[72]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_108: "f32[72]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_85: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_87: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_109: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    relu_21: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_22: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 144, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 144, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_111: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_22: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
    mul_154: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[144]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_112: "f32[144]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_157: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[144]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[144]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_113: "f32[144]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_89: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_91: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_114: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    relu_22: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    convolution_23: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_22, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 72, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 72, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_116: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_23: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
    mul_161: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[72]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_117: "f32[72]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_164: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[72]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[72]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
    add_118: "f32[72]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_93: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_95: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_119: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    relu_23: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    cat_3: "f32[8, 432, 28, 28]" = torch.ops.aten.cat.default([relu_19, relu_21, relu_23, relu_18], 1)
    convolution_24: "f32[8, 288, 28, 28]" = torch.ops.aten.convolution.default(cat_3, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_200, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 288, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 288, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_121: "f32[1, 288, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 288, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_24: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
    mul_168: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[288]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[288]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_122: "f32[288]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_171: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[288]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[288]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_123: "f32[288]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_97: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_99: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_124: "f32[8, 288, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    relu_24: "f32[8, 288, 28, 28]" = torch.ops.aten.relu.default(add_124);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_25: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 304, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 304, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_126: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_25: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
    mul_175: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[304]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_127: "f32[304]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_178: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_179: "f32[304]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[304]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_128: "f32[304]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_101: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_103: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_129: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    relu_25: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_26: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_206, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 304, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 304, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_131: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_26: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_53)
    mul_182: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[304]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_132: "f32[304]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_185: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_186: "f32[304]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[304]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_133: "f32[304]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_105: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_107: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_134: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    relu_26: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_134);  add_134 = None
    convolution_27: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_26, primals_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 152, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 152, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_136: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
    mul_189: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[152]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_137: "f32[152]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_192: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_193: "f32[152]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[152]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_138: "f32[152]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_109: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_111: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_139: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    relu_27: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_28: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_212, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 304, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 304, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_141: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
    mul_196: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[304]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_142: "f32[304]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_199: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_200: "f32[304]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[304]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_143: "f32[304]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_113: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_115: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_144: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    relu_28: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    convolution_29: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 152, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 152, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_146: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_29: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_59)
    mul_203: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[152]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_147: "f32[152]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_206: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_207: "f32[152]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[152]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_148: "f32[152]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_117: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_119: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_149: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    relu_29: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_149);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    cat_4: "f32[8, 608, 14, 14]" = torch.ops.aten.cat.default([relu_25, relu_27, relu_29], 1)
    convolution_30: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(cat_4, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_218, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 304, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 304, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_151: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_30: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
    mul_210: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[304]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_152: "f32[304]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_213: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_214: "f32[304]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[304]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_153: "f32[304]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_121: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_123: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_154: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    relu_30: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_154);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    convolution_31: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_30, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_221, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 304, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 304, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_156: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_31: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
    mul_217: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[304]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_157: "f32[304]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_221: "f32[304]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[304]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_158: "f32[304]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_125: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_127: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_159: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    relu_31: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    convolution_32: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_31, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_224, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 304, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 304, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_161: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_32: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_65)
    mul_224: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[304]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_162: "f32[304]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_227: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_228: "f32[304]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[304]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_163: "f32[304]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_129: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_131: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_164: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    relu_32: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_164);  add_164 = None
    convolution_33: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_227, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 152, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 152, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_166: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_33: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
    mul_231: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[152]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_167: "f32[152]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_235: "f32[152]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[152]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_168: "f32[152]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_133: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_135: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_169: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    relu_33: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_169);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    convolution_34: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_230, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 304, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 304, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_171: "f32[1, 304, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 304, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_34: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
    mul_238: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[304]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[304]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_172: "f32[304]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[304]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_242: "f32[304]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[304]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_173: "f32[304]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_137: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_139: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_174: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    relu_34: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    convolution_35: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_34, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_233, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 152, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 152, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_176: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_35: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_71)
    mul_245: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[152]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_177: "f32[152]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_249: "f32[152]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[152]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_178: "f32[152]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_141: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_143: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_179: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    relu_35: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    cat_5: "f32[8, 912, 14, 14]" = torch.ops.aten.cat.default([relu_31, relu_33, relu_35, relu_30], 1)
    convolution_36: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(cat_5, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_236, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 480, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 480, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_181: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
    mul_252: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[480]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_182: "f32[480]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_256: "f32[480]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[480]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_183: "f32[480]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_184: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    relu_36: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:171, code: x = self.head(self.from_seq(x))
    convolution_37: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_112, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 960, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 960, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_186: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_37: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
    mul_259: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[960]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_187: "f32[960]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_262: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_263: "f32[960]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[960]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_188: "f32[960]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_149: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_151: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_189: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    relu_37: "f32[8, 960, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    convolution_38: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_37, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_242, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1024, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1024, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_191: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_38: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_77)
    mul_266: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_192: "f32[1024]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_269: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_193: "f32[1024]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_194: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    relu_38: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    convolution_39: "f32[8, 1280, 4, 4]" = torch.ops.aten.convolution.default(relu_38, primals_118, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_195: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1280, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 1280, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_196: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_39: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
    mul_273: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_197: "f32[1280]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_276: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0078740157480315);  squeeze_119 = None
    mul_277: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_198: "f32[1280]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_157: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_159: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_199: "f32[8, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    relu_39: "f32[8, 1280, 4, 4]" = torch.ops.aten.relu.default(add_199);  add_199 = None
    convolution_40: "f32[8, 1024, 4, 4]" = torch.ops.aten.convolution.default(relu_39, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_200: "i64[]" = torch.ops.aten.add.Tensor(primals_248, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1024, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 1024, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_201: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_40: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
    mul_280: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_202: "f32[1024]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_283: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0078740157480315);  squeeze_122 = None
    mul_284: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_203: "f32[1024]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_161: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_163: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_204: "f32[8, 1024, 4, 4]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    relu_40: "f32[8, 1024, 4, 4]" = torch.ops.aten.relu.default(add_204);  add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_40, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.view.default(mean, [8, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:176, code: x = self.head_drop(x)
    clone: "f32[8, 1024]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:177, code: return x if pre_logits else self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_125, clone, permute);  primals_125 = None
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
    expand: "f32[8, 1024, 4, 4]" = torch.ops.aten.expand.default(view_2, [8, 1024, 4, 4]);  view_2 = None
    div: "f32[8, 1024, 4, 4]" = torch.ops.aten.div.Scalar(expand, 16);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:171, code: x = self.head(self.from_seq(x))
    alias_42: "f32[8, 1024, 4, 4]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_43: "f32[8, 1024, 4, 4]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le: "b8[8, 1024, 4, 4]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 1024, 4, 4]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    unsqueeze_164: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_165: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_41: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_166)
    mul_287: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(where, sub_41);  sub_41 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_288: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
    unsqueeze_167: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
    unsqueeze_168: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_289: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.0078125)
    mul_290: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_291: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    unsqueeze_170: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_171: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_173: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_174: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    sub_42: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_166);  convolution_40 = unsqueeze_166 = None
    mul_293: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_172);  sub_42 = unsqueeze_172 = None
    sub_43: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(where, mul_293);  where = mul_293 = None
    sub_44: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(sub_43, unsqueeze_169);  sub_43 = unsqueeze_169 = None
    mul_294: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_175);  sub_44 = unsqueeze_175 = None
    mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_121);  sum_3 = squeeze_121 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_294, relu_39, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_294 = primals_121 = None
    getitem_82: "f32[8, 1280, 4, 4]" = convolution_backward[0]
    getitem_83: "f32[1024, 1280, 1, 1]" = convolution_backward[1];  convolution_backward = None
    alias_45: "f32[8, 1280, 4, 4]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_46: "f32[8, 1280, 4, 4]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_1: "b8[8, 1280, 4, 4]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 1280, 4, 4]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_82);  le_1 = scalar_tensor_1 = getitem_82 = None
    unsqueeze_176: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_177: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    sum_4: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_45: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_178)
    mul_296: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(where_1, sub_45);  sub_45 = None
    sum_5: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 2, 3]);  mul_296 = None
    mul_297: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_4, 0.0078125)
    unsqueeze_179: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
    unsqueeze_180: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_298: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_5, 0.0078125)
    mul_299: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_300: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_182: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
    unsqueeze_183: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_301: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_185: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_186: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    sub_46: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_178);  convolution_39 = unsqueeze_178 = None
    mul_302: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_184);  sub_46 = unsqueeze_184 = None
    sub_47: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(where_1, mul_302);  where_1 = mul_302 = None
    sub_48: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(sub_47, unsqueeze_181);  sub_47 = unsqueeze_181 = None
    mul_303: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_187);  sub_48 = unsqueeze_187 = None
    mul_304: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_118);  sum_5 = squeeze_118 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_303, relu_38, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = primals_118 = None
    getitem_85: "f32[8, 1024, 7, 7]" = convolution_backward_1[0]
    getitem_86: "f32[1280, 1024, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    alias_48: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_49: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_2: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_85);  le_2 = scalar_tensor_2 = getitem_85 = None
    unsqueeze_188: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_189: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_49: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_190)
    mul_305: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_49);  sub_49 = None
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 2, 3]);  mul_305 = None
    mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_191: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
    unsqueeze_192: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_307: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_308: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_309: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    unsqueeze_194: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_309, 0);  mul_309 = None
    unsqueeze_195: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_310: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_197: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
    unsqueeze_198: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    sub_50: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_190);  convolution_38 = unsqueeze_190 = None
    mul_311: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_196);  sub_50 = unsqueeze_196 = None
    sub_51: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_311);  where_2 = mul_311 = None
    sub_52: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_193);  sub_51 = unsqueeze_193 = None
    mul_312: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_199);  sub_52 = unsqueeze_199 = None
    mul_313: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_115);  sum_7 = squeeze_115 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_312, relu_37, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_312 = primals_115 = None
    getitem_88: "f32[8, 960, 7, 7]" = convolution_backward_2[0]
    getitem_89: "f32[1024, 960, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    alias_51: "f32[8, 960, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_52: "f32[8, 960, 7, 7]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_3: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_88);  le_3 = scalar_tensor_3 = getitem_88 = None
    unsqueeze_200: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_201: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_53: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_202)
    mul_314: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_53);  sub_53 = None
    sum_9: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_315: "f32[960]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_203: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_204: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_316: "f32[960]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_317: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_318: "f32[960]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_206: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_207: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_319: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_209: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_210: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sub_54: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_202);  convolution_37 = unsqueeze_202 = None
    mul_320: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_208);  sub_54 = unsqueeze_208 = None
    sub_55: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_320);  where_3 = mul_320 = None
    sub_56: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_205);  sub_55 = unsqueeze_205 = None
    mul_321: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_211);  sub_56 = unsqueeze_211 = None
    mul_322: "f32[960]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_112);  sum_9 = squeeze_112 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_321, relu_36, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_321 = primals_112 = None
    getitem_91: "f32[8, 480, 14, 14]" = convolution_backward_3[0]
    getitem_92: "f32[960, 480, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    alias_54: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_55: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_4: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_91);  le_4 = scalar_tensor_4 = getitem_91 = None
    unsqueeze_212: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_213: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    sum_10: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_214)
    mul_323: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_57);  sub_57 = None
    sum_11: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 2, 3]);  mul_323 = None
    mul_324: "f32[480]" = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
    unsqueeze_215: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_216: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_325: "f32[480]" = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
    mul_326: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_327: "f32[480]" = torch.ops.aten.mul.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    unsqueeze_218: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_219: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_328: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_221: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_222: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    sub_58: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_214);  convolution_36 = unsqueeze_214 = None
    mul_329: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_220);  sub_58 = unsqueeze_220 = None
    sub_59: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_329);  where_4 = mul_329 = None
    sub_60: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_217);  sub_59 = unsqueeze_217 = None
    mul_330: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_223);  sub_60 = unsqueeze_223 = None
    mul_331: "f32[480]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_109);  sum_11 = squeeze_109 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_330, cat_5, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_330 = cat_5 = primals_109 = None
    getitem_94: "f32[8, 912, 14, 14]" = convolution_backward_4[0]
    getitem_95: "f32[480, 912, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    slice_1: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 0, 304)
    slice_2: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 304, 456)
    slice_3: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 456, 608)
    slice_4: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 608, 912);  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_57: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_58: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_5: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, slice_3);  le_5 = scalar_tensor_5 = slice_3 = None
    unsqueeze_224: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_225: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    sum_12: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_61: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_226)
    mul_332: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_61);  sub_61 = None
    sum_13: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 2, 3]);  mul_332 = None
    mul_333: "f32[152]" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
    unsqueeze_227: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_228: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_334: "f32[152]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    mul_335: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_336: "f32[152]" = torch.ops.aten.mul.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    unsqueeze_230: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_336, 0);  mul_336 = None
    unsqueeze_231: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_337: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_233: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
    unsqueeze_234: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    sub_62: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_226);  convolution_35 = unsqueeze_226 = None
    mul_338: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_232);  sub_62 = unsqueeze_232 = None
    sub_63: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_5, mul_338);  where_5 = mul_338 = None
    sub_64: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_229);  sub_63 = unsqueeze_229 = None
    mul_339: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_235);  sub_64 = unsqueeze_235 = None
    mul_340: "f32[152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_106);  sum_13 = squeeze_106 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_339, relu_34, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_339 = primals_106 = None
    getitem_97: "f32[8, 304, 14, 14]" = convolution_backward_5[0]
    getitem_98: "f32[152, 304, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    alias_60: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_61: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_6: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_97);  le_6 = scalar_tensor_6 = getitem_97 = None
    unsqueeze_236: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_237: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    sum_14: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_65: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_238)
    mul_341: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_65);  sub_65 = None
    sum_15: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3]);  mul_341 = None
    mul_342: "f32[304]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    unsqueeze_239: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_240: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_343: "f32[304]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
    mul_344: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_345: "f32[304]" = torch.ops.aten.mul.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
    unsqueeze_242: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_243: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_346: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_245: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_246: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    sub_66: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_238);  convolution_34 = unsqueeze_238 = None
    mul_347: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_244);  sub_66 = unsqueeze_244 = None
    sub_67: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_347);  where_6 = mul_347 = None
    sub_68: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_241);  sub_67 = unsqueeze_241 = None
    mul_348: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_247);  sub_68 = unsqueeze_247 = None
    mul_349: "f32[304]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_103);  sum_15 = squeeze_103 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_348, relu_33, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_348 = primals_103 = None
    getitem_100: "f32[8, 152, 14, 14]" = convolution_backward_6[0]
    getitem_101: "f32[304, 152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_205: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(slice_2, getitem_100);  slice_2 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_63: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_64: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_7: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, add_205);  le_7 = scalar_tensor_7 = add_205 = None
    unsqueeze_248: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_249: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    sum_16: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_69: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_250)
    mul_350: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_69);  sub_69 = None
    sum_17: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 2, 3]);  mul_350 = None
    mul_351: "f32[152]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
    unsqueeze_251: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_252: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_352: "f32[152]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_353: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_354: "f32[152]" = torch.ops.aten.mul.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    unsqueeze_254: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_255: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_355: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_257: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_258: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sub_70: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_250);  convolution_33 = unsqueeze_250 = None
    mul_356: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_256);  sub_70 = unsqueeze_256 = None
    sub_71: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_356);  where_7 = mul_356 = None
    sub_72: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_253);  sub_71 = unsqueeze_253 = None
    mul_357: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_259);  sub_72 = unsqueeze_259 = None
    mul_358: "f32[152]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_100);  sum_17 = squeeze_100 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_357, relu_32, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_357 = primals_100 = None
    getitem_103: "f32[8, 304, 14, 14]" = convolution_backward_7[0]
    getitem_104: "f32[152, 304, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    alias_66: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_67: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_8: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_103);  le_8 = scalar_tensor_8 = getitem_103 = None
    unsqueeze_260: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_261: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_18: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_73: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_262)
    mul_359: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_73);  sub_73 = None
    sum_19: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 2, 3]);  mul_359 = None
    mul_360: "f32[304]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_263: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
    unsqueeze_264: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_361: "f32[304]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_362: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_363: "f32[304]" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_266: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
    unsqueeze_267: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_364: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_269: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
    unsqueeze_270: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_74: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_262);  convolution_32 = unsqueeze_262 = None
    mul_365: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_268);  sub_74 = unsqueeze_268 = None
    sub_75: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_365);  where_8 = mul_365 = None
    sub_76: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_265);  sub_75 = unsqueeze_265 = None
    mul_366: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_271);  sub_76 = unsqueeze_271 = None
    mul_367: "f32[304]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_97);  sum_19 = squeeze_97 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_366, relu_31, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_366 = primals_97 = None
    getitem_106: "f32[8, 304, 14, 14]" = convolution_backward_8[0]
    getitem_107: "f32[304, 304, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_206: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_1, getitem_106);  slice_1 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_69: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_70: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_9: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_206);  le_9 = scalar_tensor_9 = add_206 = None
    unsqueeze_272: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_273: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_20: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_77: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_274)
    mul_368: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_77);  sub_77 = None
    sum_21: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3]);  mul_368 = None
    mul_369: "f32[304]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_275: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_276: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_370: "f32[304]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_371: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_372: "f32[304]" = torch.ops.aten.mul.Tensor(mul_370, mul_371);  mul_370 = mul_371 = None
    unsqueeze_278: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_279: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_373: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_281: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
    unsqueeze_282: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_78: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_274);  convolution_31 = unsqueeze_274 = None
    mul_374: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_280);  sub_78 = unsqueeze_280 = None
    sub_79: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_374);  where_9 = mul_374 = None
    sub_80: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_277);  sub_79 = unsqueeze_277 = None
    mul_375: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_283);  sub_80 = unsqueeze_283 = None
    mul_376: "f32[304]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_94);  sum_21 = squeeze_94 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_375, relu_30, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_375 = primals_94 = None
    getitem_109: "f32[8, 304, 14, 14]" = convolution_backward_9[0]
    getitem_110: "f32[304, 304, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_207: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_4, getitem_109);  slice_4 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    alias_72: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_73: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_10: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_207);  le_10 = scalar_tensor_10 = add_207 = None
    unsqueeze_284: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_285: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_22: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_81: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_286)
    mul_377: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_81);  sub_81 = None
    sum_23: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
    mul_378: "f32[304]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_287: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_288: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_379: "f32[304]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_380: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_381: "f32[304]" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_290: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
    unsqueeze_291: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_382: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_293: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_294: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_82: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_286);  convolution_30 = unsqueeze_286 = None
    mul_383: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_292);  sub_82 = unsqueeze_292 = None
    sub_83: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_383);  where_10 = mul_383 = None
    sub_84: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_289);  sub_83 = unsqueeze_289 = None
    mul_384: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_295);  sub_84 = unsqueeze_295 = None
    mul_385: "f32[304]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_91);  sum_23 = squeeze_91 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_384, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_384 = cat_4 = primals_91 = None
    getitem_112: "f32[8, 608, 14, 14]" = convolution_backward_10[0]
    getitem_113: "f32[304, 608, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    slice_5: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 0, 304)
    slice_6: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 304, 456)
    slice_7: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 456, 608);  getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_75: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_76: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_11: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, slice_7);  le_11 = scalar_tensor_11 = slice_7 = None
    unsqueeze_296: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_297: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_24: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_85: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_298)
    mul_386: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_85);  sub_85 = None
    sum_25: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 2, 3]);  mul_386 = None
    mul_387: "f32[152]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_299: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_300: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_388: "f32[152]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_389: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_390: "f32[152]" = torch.ops.aten.mul.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    unsqueeze_302: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
    unsqueeze_303: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_391: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_305: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
    unsqueeze_306: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_86: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_298);  convolution_29 = unsqueeze_298 = None
    mul_392: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_304);  sub_86 = unsqueeze_304 = None
    sub_87: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_392);  where_11 = mul_392 = None
    sub_88: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_301);  sub_87 = unsqueeze_301 = None
    mul_393: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_307);  sub_88 = unsqueeze_307 = None
    mul_394: "f32[152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_88);  sum_25 = squeeze_88 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_393, relu_28, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_393 = primals_88 = None
    getitem_115: "f32[8, 304, 14, 14]" = convolution_backward_11[0]
    getitem_116: "f32[152, 304, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    alias_78: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_79: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_12: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, getitem_115);  le_12 = scalar_tensor_12 = getitem_115 = None
    unsqueeze_308: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_309: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_26: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_89: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_310)
    mul_395: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_89);  sub_89 = None
    sum_27: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3]);  mul_395 = None
    mul_396: "f32[304]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_311: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_312: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_397: "f32[304]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_398: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_399: "f32[304]" = torch.ops.aten.mul.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    unsqueeze_314: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
    unsqueeze_315: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_400: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_317: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_318: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_90: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_310);  convolution_28 = unsqueeze_310 = None
    mul_401: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_316);  sub_90 = unsqueeze_316 = None
    sub_91: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_401);  where_12 = mul_401 = None
    sub_92: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_313);  sub_91 = unsqueeze_313 = None
    mul_402: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_319);  sub_92 = unsqueeze_319 = None
    mul_403: "f32[304]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_85);  sum_27 = squeeze_85 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_402, relu_27, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_402 = primals_85 = None
    getitem_118: "f32[8, 152, 14, 14]" = convolution_backward_12[0]
    getitem_119: "f32[304, 152, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_208: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(slice_6, getitem_118);  slice_6 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_81: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_82: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_13: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, add_208);  le_13 = scalar_tensor_13 = add_208 = None
    unsqueeze_320: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_321: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_28: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_93: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_322)
    mul_404: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_93);  sub_93 = None
    sum_29: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3]);  mul_404 = None
    mul_405: "f32[152]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_323: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_324: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_406: "f32[152]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_407: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_408: "f32[152]" = torch.ops.aten.mul.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    unsqueeze_326: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_327: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_409: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_329: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_330: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_94: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_322);  convolution_27 = unsqueeze_322 = None
    mul_410: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_328);  sub_94 = unsqueeze_328 = None
    sub_95: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_410);  where_13 = mul_410 = None
    sub_96: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_325);  sub_95 = unsqueeze_325 = None
    mul_411: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_331);  sub_96 = unsqueeze_331 = None
    mul_412: "f32[152]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_82);  sum_29 = squeeze_82 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_411, relu_26, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_411 = primals_82 = None
    getitem_121: "f32[8, 304, 14, 14]" = convolution_backward_13[0]
    getitem_122: "f32[152, 304, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    alias_84: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_85: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_14: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_121);  le_14 = scalar_tensor_14 = getitem_121 = None
    unsqueeze_332: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_333: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_30: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_97: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_334)
    mul_413: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_97);  sub_97 = None
    sum_31: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 2, 3]);  mul_413 = None
    mul_414: "f32[304]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_335: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_336: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_415: "f32[304]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_416: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_417: "f32[304]" = torch.ops.aten.mul.Tensor(mul_415, mul_416);  mul_415 = mul_416 = None
    unsqueeze_338: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_339: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_418: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_341: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_342: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_98: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_334);  convolution_26 = unsqueeze_334 = None
    mul_419: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_340);  sub_98 = unsqueeze_340 = None
    sub_99: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_419);  where_14 = mul_419 = None
    sub_100: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_337);  sub_99 = unsqueeze_337 = None
    mul_420: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_343);  sub_100 = unsqueeze_343 = None
    mul_421: "f32[304]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_79);  sum_31 = squeeze_79 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_420, relu_25, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_420 = primals_79 = None
    getitem_124: "f32[8, 304, 14, 14]" = convolution_backward_14[0]
    getitem_125: "f32[304, 304, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_209: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, getitem_124);  slice_5 = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_87: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_88: "f32[8, 304, 14, 14]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_15: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_209);  le_15 = scalar_tensor_15 = add_209 = None
    unsqueeze_344: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_345: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_32: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_101: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_346)
    mul_422: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_101);  sub_101 = None
    sum_33: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3]);  mul_422 = None
    mul_423: "f32[304]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_347: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_348: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_424: "f32[304]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_425: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_426: "f32[304]" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_350: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_351: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_427: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_353: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
    unsqueeze_354: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_102: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_346);  convolution_25 = unsqueeze_346 = None
    mul_428: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_352);  sub_102 = unsqueeze_352 = None
    sub_103: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_428);  where_15 = mul_428 = None
    sub_104: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_349);  sub_103 = unsqueeze_349 = None
    mul_429: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_355);  sub_104 = unsqueeze_355 = None
    mul_430: "f32[304]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_76);  sum_33 = squeeze_76 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_429, relu_24, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_429 = primals_76 = None
    getitem_127: "f32[8, 288, 28, 28]" = convolution_backward_15[0]
    getitem_128: "f32[304, 288, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    alias_90: "f32[8, 288, 28, 28]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_91: "f32[8, 288, 28, 28]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_16: "b8[8, 288, 28, 28]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 288, 28, 28]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_127);  le_16 = scalar_tensor_16 = getitem_127 = None
    unsqueeze_356: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_357: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_34: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_105: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_358)
    mul_431: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_105);  sub_105 = None
    sum_35: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
    mul_432: "f32[288]" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
    unsqueeze_359: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_360: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_433: "f32[288]" = torch.ops.aten.mul.Tensor(sum_35, 0.00015943877551020407)
    mul_434: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_435: "f32[288]" = torch.ops.aten.mul.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    unsqueeze_362: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_363: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_436: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_365: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_366: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_106: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_358);  convolution_24 = unsqueeze_358 = None
    mul_437: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_364);  sub_106 = unsqueeze_364 = None
    sub_107: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_437);  where_16 = mul_437 = None
    sub_108: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_361);  sub_107 = unsqueeze_361 = None
    mul_438: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_367);  sub_108 = unsqueeze_367 = None
    mul_439: "f32[288]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_73);  sum_35 = squeeze_73 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_438, cat_3, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_438 = cat_3 = primals_73 = None
    getitem_130: "f32[8, 432, 28, 28]" = convolution_backward_16[0]
    getitem_131: "f32[288, 432, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    slice_8: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 0, 144)
    slice_9: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 144, 216)
    slice_10: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 216, 288)
    slice_11: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 288, 432);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_93: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_94: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_17: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, slice_10);  le_17 = scalar_tensor_17 = slice_10 = None
    unsqueeze_368: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_369: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_36: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_109: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_370)
    mul_440: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_109);  sub_109 = None
    sum_37: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 2, 3]);  mul_440 = None
    mul_441: "f32[72]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
    unsqueeze_371: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_372: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_442: "f32[72]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
    mul_443: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_444: "f32[72]" = torch.ops.aten.mul.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    unsqueeze_374: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_375: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_445: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_377: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
    unsqueeze_378: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    sub_110: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_370);  convolution_23 = unsqueeze_370 = None
    mul_446: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_376);  sub_110 = unsqueeze_376 = None
    sub_111: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_446);  where_17 = mul_446 = None
    sub_112: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_373);  sub_111 = unsqueeze_373 = None
    mul_447: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_379);  sub_112 = unsqueeze_379 = None
    mul_448: "f32[72]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_70);  sum_37 = squeeze_70 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_447, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_447 = primals_70 = None
    getitem_133: "f32[8, 144, 28, 28]" = convolution_backward_17[0]
    getitem_134: "f32[72, 144, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    alias_96: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_97: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_18: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, getitem_133);  le_18 = scalar_tensor_18 = getitem_133 = None
    unsqueeze_380: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_381: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_38: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_113: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_382)
    mul_449: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_18, sub_113);  sub_113 = None
    sum_39: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
    mul_450: "f32[144]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
    unsqueeze_383: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_384: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_451: "f32[144]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
    mul_452: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_453: "f32[144]" = torch.ops.aten.mul.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
    unsqueeze_386: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
    unsqueeze_387: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_454: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_389: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_390: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    sub_114: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_382);  convolution_22 = unsqueeze_382 = None
    mul_455: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_388);  sub_114 = unsqueeze_388 = None
    sub_115: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_18, mul_455);  where_18 = mul_455 = None
    sub_116: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_385);  sub_115 = unsqueeze_385 = None
    mul_456: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_391);  sub_116 = unsqueeze_391 = None
    mul_457: "f32[144]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_67);  sum_39 = squeeze_67 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_456, relu_21, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_456 = primals_67 = None
    getitem_136: "f32[8, 72, 28, 28]" = convolution_backward_18[0]
    getitem_137: "f32[144, 72, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_210: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(slice_9, getitem_136);  slice_9 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_99: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_100: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_19: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, add_210);  le_19 = scalar_tensor_19 = add_210 = None
    unsqueeze_392: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_393: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_40: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_117: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_394)
    mul_458: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_19, sub_117);  sub_117 = None
    sum_41: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[72]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
    unsqueeze_395: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_396: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_460: "f32[72]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
    mul_461: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_462: "f32[72]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_398: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_399: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_463: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_401: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_402: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    sub_118: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_394);  convolution_21 = unsqueeze_394 = None
    mul_464: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_400);  sub_118 = unsqueeze_400 = None
    sub_119: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_19, mul_464);  where_19 = mul_464 = None
    sub_120: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_397);  sub_119 = unsqueeze_397 = None
    mul_465: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_403);  sub_120 = unsqueeze_403 = None
    mul_466: "f32[72]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_64);  sum_41 = squeeze_64 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_465, relu_20, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = primals_64 = None
    getitem_139: "f32[8, 144, 28, 28]" = convolution_backward_19[0]
    getitem_140: "f32[72, 144, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    alias_102: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_103: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_20: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_139);  le_20 = scalar_tensor_20 = getitem_139 = None
    unsqueeze_404: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_405: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_42: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_121: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_406)
    mul_467: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_121);  sub_121 = None
    sum_43: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[144]" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
    unsqueeze_407: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_408: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_469: "f32[144]" = torch.ops.aten.mul.Tensor(sum_43, 0.00015943877551020407)
    mul_470: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_471: "f32[144]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_410: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_411: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_472: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_413: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_414: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    sub_122: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_406);  convolution_20 = unsqueeze_406 = None
    mul_473: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_412);  sub_122 = unsqueeze_412 = None
    sub_123: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_20, mul_473);  where_20 = mul_473 = None
    sub_124: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_409);  sub_123 = unsqueeze_409 = None
    mul_474: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_415);  sub_124 = unsqueeze_415 = None
    mul_475: "f32[144]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_61);  sum_43 = squeeze_61 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_474, relu_19, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = primals_61 = None
    getitem_142: "f32[8, 144, 28, 28]" = convolution_backward_20[0]
    getitem_143: "f32[144, 144, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_211: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_8, getitem_142);  slice_8 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_105: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_106: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_21: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_211);  le_21 = scalar_tensor_21 = add_211 = None
    unsqueeze_416: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_417: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_44: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_125: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_418)
    mul_476: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_125);  sub_125 = None
    sum_45: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_477: "f32[144]" = torch.ops.aten.mul.Tensor(sum_44, 0.00015943877551020407)
    unsqueeze_419: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_420: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_478: "f32[144]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
    mul_479: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_480: "f32[144]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_422: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_423: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_481: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_425: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_426: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_126: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_418);  convolution_19 = unsqueeze_418 = None
    mul_482: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_424);  sub_126 = unsqueeze_424 = None
    sub_127: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_21, mul_482);  where_21 = mul_482 = None
    sub_128: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_421);  sub_127 = unsqueeze_421 = None
    mul_483: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_427);  sub_128 = unsqueeze_427 = None
    mul_484: "f32[144]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_58);  sum_45 = squeeze_58 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_483, relu_18, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_483 = primals_58 = None
    getitem_145: "f32[8, 144, 28, 28]" = convolution_backward_21[0]
    getitem_146: "f32[144, 144, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_212: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_11, getitem_145);  slice_11 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    alias_108: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_109: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_22: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_212);  le_22 = scalar_tensor_22 = add_212 = None
    unsqueeze_428: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_429: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_46: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_129: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_430)
    mul_485: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_129);  sub_129 = None
    sum_47: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3]);  mul_485 = None
    mul_486: "f32[144]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
    unsqueeze_431: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_432: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_487: "f32[144]" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
    mul_488: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_489: "f32[144]" = torch.ops.aten.mul.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_434: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
    unsqueeze_435: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_490: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_437: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_438: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_130: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_430);  convolution_18 = unsqueeze_430 = None
    mul_491: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_436);  sub_130 = unsqueeze_436 = None
    sub_131: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_22, mul_491);  where_22 = mul_491 = None
    sub_132: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_433);  sub_131 = unsqueeze_433 = None
    mul_492: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_439);  sub_132 = unsqueeze_439 = None
    mul_493: "f32[144]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_55);  sum_47 = squeeze_55 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_492, cat_2, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_492 = cat_2 = primals_55 = None
    getitem_148: "f32[8, 288, 28, 28]" = convolution_backward_22[0]
    getitem_149: "f32[144, 288, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    slice_12: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 0, 144)
    slice_13: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 144, 216)
    slice_14: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 216, 288);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_111: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_112: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_23: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, slice_14);  le_23 = scalar_tensor_23 = slice_14 = None
    unsqueeze_440: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_441: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_48: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_133: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_442)
    mul_494: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_133);  sub_133 = None
    sum_49: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3]);  mul_494 = None
    mul_495: "f32[72]" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
    unsqueeze_443: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_444: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_496: "f32[72]" = torch.ops.aten.mul.Tensor(sum_49, 0.00015943877551020407)
    mul_497: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_498: "f32[72]" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
    unsqueeze_446: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    unsqueeze_447: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_499: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_449: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_450: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_134: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_442);  convolution_17 = unsqueeze_442 = None
    mul_500: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_448);  sub_134 = unsqueeze_448 = None
    sub_135: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_23, mul_500);  where_23 = mul_500 = None
    sub_136: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_445);  sub_135 = unsqueeze_445 = None
    mul_501: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_451);  sub_136 = unsqueeze_451 = None
    mul_502: "f32[72]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_52);  sum_49 = squeeze_52 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_501, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_501 = primals_52 = None
    getitem_151: "f32[8, 144, 28, 28]" = convolution_backward_23[0]
    getitem_152: "f32[72, 144, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    alias_114: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_115: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_24: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, getitem_151);  le_24 = scalar_tensor_24 = getitem_151 = None
    unsqueeze_452: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_453: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_50: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_137: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_454)
    mul_503: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_137);  sub_137 = None
    sum_51: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_504: "f32[144]" = torch.ops.aten.mul.Tensor(sum_50, 0.00015943877551020407)
    unsqueeze_455: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_456: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_505: "f32[144]" = torch.ops.aten.mul.Tensor(sum_51, 0.00015943877551020407)
    mul_506: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_507: "f32[144]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_458: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_459: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_508: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_461: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_462: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_138: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_454);  convolution_16 = unsqueeze_454 = None
    mul_509: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_460);  sub_138 = unsqueeze_460 = None
    sub_139: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_509);  where_24 = mul_509 = None
    sub_140: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_457);  sub_139 = unsqueeze_457 = None
    mul_510: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_463);  sub_140 = unsqueeze_463 = None
    mul_511: "f32[144]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_49);  sum_51 = squeeze_49 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_510, relu_15, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_510 = primals_49 = None
    getitem_154: "f32[8, 72, 28, 28]" = convolution_backward_24[0]
    getitem_155: "f32[144, 72, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_213: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(slice_13, getitem_154);  slice_13 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_117: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_118: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_25: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, add_213);  le_25 = scalar_tensor_25 = add_213 = None
    unsqueeze_464: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_465: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_52: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_141: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466)
    mul_512: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_141);  sub_141 = None
    sum_53: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
    mul_513: "f32[72]" = torch.ops.aten.mul.Tensor(sum_52, 0.00015943877551020407)
    unsqueeze_467: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_468: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_514: "f32[72]" = torch.ops.aten.mul.Tensor(sum_53, 0.00015943877551020407)
    mul_515: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_516: "f32[72]" = torch.ops.aten.mul.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    unsqueeze_470: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_471: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_517: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_473: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_474: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_142: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466);  convolution_15 = unsqueeze_466 = None
    mul_518: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_472);  sub_142 = unsqueeze_472 = None
    sub_143: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_518);  where_25 = mul_518 = None
    sub_144: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_469);  sub_143 = unsqueeze_469 = None
    mul_519: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_475);  sub_144 = unsqueeze_475 = None
    mul_520: "f32[72]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_46);  sum_53 = squeeze_46 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_519, relu_14, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_519 = primals_46 = None
    getitem_157: "f32[8, 144, 28, 28]" = convolution_backward_25[0]
    getitem_158: "f32[72, 144, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    alias_120: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_121: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_26: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_157);  le_26 = scalar_tensor_26 = getitem_157 = None
    unsqueeze_476: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_477: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_54: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_145: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478)
    mul_521: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_145);  sub_145 = None
    sum_55: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3]);  mul_521 = None
    mul_522: "f32[144]" = torch.ops.aten.mul.Tensor(sum_54, 0.00015943877551020407)
    unsqueeze_479: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_480: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_523: "f32[144]" = torch.ops.aten.mul.Tensor(sum_55, 0.00015943877551020407)
    mul_524: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_525: "f32[144]" = torch.ops.aten.mul.Tensor(mul_523, mul_524);  mul_523 = mul_524 = None
    unsqueeze_482: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    unsqueeze_483: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_526: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_485: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_486: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_146: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478);  convolution_14 = unsqueeze_478 = None
    mul_527: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_484);  sub_146 = unsqueeze_484 = None
    sub_147: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_527);  where_26 = mul_527 = None
    sub_148: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_481);  sub_147 = unsqueeze_481 = None
    mul_528: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_487);  sub_148 = unsqueeze_487 = None
    mul_529: "f32[144]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_43);  sum_55 = squeeze_43 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_528, relu_13, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_528 = primals_43 = None
    getitem_160: "f32[8, 144, 28, 28]" = convolution_backward_26[0]
    getitem_161: "f32[144, 144, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_214: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_12, getitem_160);  slice_12 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_123: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_124: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_27: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, add_214);  le_27 = scalar_tensor_27 = add_214 = None
    unsqueeze_488: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_489: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_56: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_149: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490)
    mul_530: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_149);  sub_149 = None
    sum_57: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 2, 3]);  mul_530 = None
    mul_531: "f32[144]" = torch.ops.aten.mul.Tensor(sum_56, 0.00015943877551020407)
    unsqueeze_491: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_492: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_532: "f32[144]" = torch.ops.aten.mul.Tensor(sum_57, 0.00015943877551020407)
    mul_533: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_534: "f32[144]" = torch.ops.aten.mul.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_494: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_495: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_535: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_497: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    unsqueeze_498: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_150: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490);  convolution_13 = unsqueeze_490 = None
    mul_536: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_496);  sub_150 = unsqueeze_496 = None
    sub_151: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_536);  where_27 = mul_536 = None
    sub_152: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_493);  sub_151 = unsqueeze_493 = None
    mul_537: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_499);  sub_152 = unsqueeze_499 = None
    mul_538: "f32[144]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_40);  sum_57 = squeeze_40 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_537, relu_12, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_537 = primals_40 = None
    getitem_163: "f32[8, 128, 56, 56]" = convolution_backward_27[0]
    getitem_164: "f32[144, 128, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    alias_126: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_127: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_28: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_163);  le_28 = scalar_tensor_28 = getitem_163 = None
    unsqueeze_500: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_501: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_153: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_502)
    mul_539: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_153);  sub_153 = None
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
    mul_540: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
    unsqueeze_503: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_504: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_541: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
    mul_542: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_543: "f32[128]" = torch.ops.aten.mul.Tensor(mul_541, mul_542);  mul_541 = mul_542 = None
    unsqueeze_506: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_507: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_544: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_509: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_510: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_154: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_502);  convolution_12 = unsqueeze_502 = None
    mul_545: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_508);  sub_154 = unsqueeze_508 = None
    sub_155: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_28, mul_545);  where_28 = mul_545 = None
    sub_156: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_505);  sub_155 = unsqueeze_505 = None
    mul_546: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_511);  sub_156 = unsqueeze_511 = None
    mul_547: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_37);  sum_59 = squeeze_37 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_546, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_546 = cat_1 = primals_37 = None
    getitem_166: "f32[8, 192, 56, 56]" = convolution_backward_28[0]
    getitem_167: "f32[128, 192, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    slice_15: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 0, 64)
    slice_16: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 64, 96)
    slice_17: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 96, 128)
    slice_18: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 128, 192);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_129: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_130: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_29: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, slice_17);  le_29 = scalar_tensor_29 = slice_17 = None
    unsqueeze_512: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_513: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_60: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_157: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_514)
    mul_548: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_157);  sub_157 = None
    sum_61: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_549: "f32[32]" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
    unsqueeze_515: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_516: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_550: "f32[32]" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
    mul_551: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_552: "f32[32]" = torch.ops.aten.mul.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_518: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_519: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_553: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_521: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_522: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    sub_158: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_514);  convolution_11 = unsqueeze_514 = None
    mul_554: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_520);  sub_158 = unsqueeze_520 = None
    sub_159: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_29, mul_554);  where_29 = mul_554 = None
    sub_160: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_517);  sub_159 = unsqueeze_517 = None
    mul_555: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_523);  sub_160 = unsqueeze_523 = None
    mul_556: "f32[32]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_34);  sum_61 = squeeze_34 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_555, relu_10, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_555 = primals_34 = None
    getitem_169: "f32[8, 64, 56, 56]" = convolution_backward_29[0]
    getitem_170: "f32[32, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    alias_132: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_133: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_30: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_169);  le_30 = scalar_tensor_30 = getitem_169 = None
    unsqueeze_524: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_525: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_62: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_161: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_526)
    mul_557: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_161);  sub_161 = None
    sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_558: "f32[64]" = torch.ops.aten.mul.Tensor(sum_62, 3.985969387755102e-05)
    unsqueeze_527: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_528: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_559: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
    mul_560: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_561: "f32[64]" = torch.ops.aten.mul.Tensor(mul_559, mul_560);  mul_559 = mul_560 = None
    unsqueeze_530: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_531: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_562: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_533: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_534: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    sub_162: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_526);  convolution_10 = unsqueeze_526 = None
    mul_563: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_532);  sub_162 = unsqueeze_532 = None
    sub_163: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_563);  where_30 = mul_563 = None
    sub_164: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_529);  sub_163 = unsqueeze_529 = None
    mul_564: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_535);  sub_164 = unsqueeze_535 = None
    mul_565: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_31);  sum_63 = squeeze_31 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_564, relu_9, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_564 = primals_31 = None
    getitem_172: "f32[8, 32, 56, 56]" = convolution_backward_30[0]
    getitem_173: "f32[64, 32, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_215: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_16, getitem_172);  slice_16 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_135: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_136: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_31: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, add_215);  le_31 = scalar_tensor_31 = add_215 = None
    unsqueeze_536: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_537: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_64: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_165: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_538)
    mul_566: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_165);  sub_165 = None
    sum_65: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3]);  mul_566 = None
    mul_567: "f32[32]" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
    unsqueeze_539: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_540: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_568: "f32[32]" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
    mul_569: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_570: "f32[32]" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    unsqueeze_542: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_543: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_571: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_545: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_546: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    sub_166: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_538);  convolution_9 = unsqueeze_538 = None
    mul_572: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_544);  sub_166 = unsqueeze_544 = None
    sub_167: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_572);  where_31 = mul_572 = None
    sub_168: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_541);  sub_167 = unsqueeze_541 = None
    mul_573: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_547);  sub_168 = unsqueeze_547 = None
    mul_574: "f32[32]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_28);  sum_65 = squeeze_28 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_573, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_573 = primals_28 = None
    getitem_175: "f32[8, 64, 56, 56]" = convolution_backward_31[0]
    getitem_176: "f32[32, 64, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    alias_138: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_139: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_32: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_175);  le_32 = scalar_tensor_32 = getitem_175 = None
    unsqueeze_548: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_549: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_66: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_169: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_550)
    mul_575: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_32, sub_169);  sub_169 = None
    sum_67: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_576: "f32[64]" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
    unsqueeze_551: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_552: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_577: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
    mul_578: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_579: "f32[64]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_554: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_555: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_580: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_557: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_558: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    sub_170: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_550);  convolution_8 = unsqueeze_550 = None
    mul_581: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_556);  sub_170 = unsqueeze_556 = None
    sub_171: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_32, mul_581);  where_32 = mul_581 = None
    sub_172: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_553);  sub_171 = unsqueeze_553 = None
    mul_582: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_559);  sub_172 = unsqueeze_559 = None
    mul_583: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_25);  sum_67 = squeeze_25 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_582, relu_7, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_582 = primals_25 = None
    getitem_178: "f32[8, 64, 56, 56]" = convolution_backward_32[0]
    getitem_179: "f32[64, 64, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_216: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_15, getitem_178);  slice_15 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_141: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_142: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_33: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, add_216);  le_33 = scalar_tensor_33 = add_216 = None
    unsqueeze_560: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_561: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_68: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_173: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_562)
    mul_584: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_33, sub_173);  sub_173 = None
    sum_69: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_585: "f32[64]" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
    unsqueeze_563: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_564: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_586: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, 3.985969387755102e-05)
    mul_587: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_588: "f32[64]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_566: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_567: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_589: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_569: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_570: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sub_174: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_562);  convolution_7 = unsqueeze_562 = None
    mul_590: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_568);  sub_174 = unsqueeze_568 = None
    sub_175: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_33, mul_590);  where_33 = mul_590 = None
    sub_176: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_565);  sub_175 = unsqueeze_565 = None
    mul_591: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_571);  sub_176 = unsqueeze_571 = None
    mul_592: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_22);  sum_69 = squeeze_22 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_591, relu_6, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = primals_22 = None
    getitem_181: "f32[8, 64, 56, 56]" = convolution_backward_33[0]
    getitem_182: "f32[64, 64, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_217: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_18, getitem_181);  slice_18 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    alias_144: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_145: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_144);  alias_144 = None
    le_34: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, add_217);  le_34 = scalar_tensor_34 = add_217 = None
    unsqueeze_572: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_573: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_177: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_574)
    mul_593: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_177);  sub_177 = None
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3]);  mul_593 = None
    mul_594: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 3.985969387755102e-05)
    unsqueeze_575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_595: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 3.985969387755102e-05)
    mul_596: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_597: "f32[64]" = torch.ops.aten.mul.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
    unsqueeze_578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_598: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    sub_178: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_574);  convolution_6 = unsqueeze_574 = None
    mul_599: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_580);  sub_178 = unsqueeze_580 = None
    sub_179: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_599);  where_34 = mul_599 = None
    sub_180: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_577);  sub_179 = unsqueeze_577 = None
    mul_600: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_583);  sub_180 = unsqueeze_583 = None
    mul_601: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_19);  sum_71 = squeeze_19 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_600, cat, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_600 = cat = primals_19 = None
    getitem_184: "f32[8, 128, 56, 56]" = convolution_backward_34[0]
    getitem_185: "f32[64, 128, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    slice_19: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 0, 64)
    slice_20: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 64, 96)
    slice_21: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 96, 128);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    alias_147: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_148: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_147);  alias_147 = None
    le_35: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_148, 0);  alias_148 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, slice_21);  le_35 = scalar_tensor_35 = slice_21 = None
    unsqueeze_584: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_585: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_72: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_181: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_586)
    mul_602: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_181);  sub_181 = None
    sum_73: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2, 3]);  mul_602 = None
    mul_603: "f32[32]" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
    unsqueeze_587: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_588: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_604: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
    mul_605: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_606: "f32[32]" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    unsqueeze_590: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
    unsqueeze_591: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_607: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_593: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    unsqueeze_594: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    sub_182: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_586);  convolution_5 = unsqueeze_586 = None
    mul_608: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_592);  sub_182 = unsqueeze_592 = None
    sub_183: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_608);  where_35 = mul_608 = None
    sub_184: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_589);  sub_183 = unsqueeze_589 = None
    mul_609: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_595);  sub_184 = unsqueeze_595 = None
    mul_610: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_16);  sum_73 = squeeze_16 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_609, relu_4, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_609 = primals_16 = None
    getitem_187: "f32[8, 64, 56, 56]" = convolution_backward_35[0]
    getitem_188: "f32[32, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    alias_150: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_151: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_150);  alias_150 = None
    le_36: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, getitem_187);  le_36 = scalar_tensor_36 = getitem_187 = None
    unsqueeze_596: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_597: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_74: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_185: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_598)
    mul_611: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_185);  sub_185 = None
    sum_75: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_612: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
    unsqueeze_599: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_600: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_613: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
    mul_614: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_615: "f32[64]" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_602: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
    unsqueeze_603: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_616: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_605: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_606: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    sub_186: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_598);  convolution_4 = unsqueeze_598 = None
    mul_617: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_604);  sub_186 = unsqueeze_604 = None
    sub_187: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_617);  where_36 = mul_617 = None
    sub_188: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_601);  sub_187 = unsqueeze_601 = None
    mul_618: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_607);  sub_188 = unsqueeze_607 = None
    mul_619: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_13);  sum_75 = squeeze_13 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_618, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_618 = primals_13 = None
    getitem_190: "f32[8, 32, 56, 56]" = convolution_backward_36[0]
    getitem_191: "f32[64, 32, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_218: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_20, getitem_190);  slice_20 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    alias_153: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_154: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_153);  alias_153 = None
    le_37: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, add_218);  le_37 = scalar_tensor_37 = add_218 = None
    unsqueeze_608: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_609: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_76: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_189: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_610)
    mul_620: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_37, sub_189);  sub_189 = None
    sum_77: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
    mul_621: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
    unsqueeze_611: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_612: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_622: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
    mul_623: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_624: "f32[32]" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    unsqueeze_614: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_615: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_625: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_617: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_618: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    sub_190: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_610);  convolution_3 = unsqueeze_610 = None
    mul_626: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_616);  sub_190 = unsqueeze_616 = None
    sub_191: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_37, mul_626);  where_37 = mul_626 = None
    sub_192: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_613);  sub_191 = unsqueeze_613 = None
    mul_627: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_619);  sub_192 = unsqueeze_619 = None
    mul_628: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_10);  sum_77 = squeeze_10 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_627, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = primals_10 = None
    getitem_193: "f32[8, 64, 56, 56]" = convolution_backward_37[0]
    getitem_194: "f32[32, 64, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    alias_156: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_157: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    le_38: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_38: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_193);  le_38 = scalar_tensor_38 = getitem_193 = None
    unsqueeze_620: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_621: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_78: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_193: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_622)
    mul_629: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_193);  sub_193 = None
    sum_79: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3]);  mul_629 = None
    mul_630: "f32[64]" = torch.ops.aten.mul.Tensor(sum_78, 3.985969387755102e-05)
    unsqueeze_623: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_624: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_631: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, 3.985969387755102e-05)
    mul_632: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_633: "f32[64]" = torch.ops.aten.mul.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    unsqueeze_626: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_627: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_634: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_629: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
    unsqueeze_630: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    sub_194: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_622);  convolution_2 = unsqueeze_622 = None
    mul_635: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_628);  sub_194 = unsqueeze_628 = None
    sub_195: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_38, mul_635);  where_38 = mul_635 = None
    sub_196: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_625);  sub_195 = unsqueeze_625 = None
    mul_636: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_631);  sub_196 = unsqueeze_631 = None
    mul_637: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_7);  sum_79 = squeeze_7 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_636, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_636 = primals_7 = None
    getitem_196: "f32[8, 64, 56, 56]" = convolution_backward_38[0]
    getitem_197: "f32[64, 64, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_219: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_19, getitem_196);  slice_19 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    alias_159: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_160: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_159);  alias_159 = None
    le_39: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, add_219);  le_39 = scalar_tensor_39 = add_219 = None
    unsqueeze_632: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_633: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_80: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_197: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_634)
    mul_638: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_197);  sub_197 = None
    sum_81: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3]);  mul_638 = None
    mul_639: "f32[64]" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
    unsqueeze_635: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_636: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_640: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
    mul_641: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_642: "f32[64]" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
    unsqueeze_638: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_639: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_643: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_641: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_642: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    sub_198: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_634);  convolution_1 = unsqueeze_634 = None
    mul_644: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_640);  sub_198 = unsqueeze_640 = None
    sub_199: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_39, mul_644);  where_39 = mul_644 = None
    sub_200: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_637);  sub_199 = unsqueeze_637 = None
    mul_645: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_643);  sub_200 = unsqueeze_643 = None
    mul_646: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_4);  sum_81 = squeeze_4 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_645, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_645 = primals_4 = None
    getitem_199: "f32[8, 32, 112, 112]" = convolution_backward_39[0]
    getitem_200: "f32[64, 32, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:169, code: x = self.stem(x)
    alias_162: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_163: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_162);  alias_162 = None
    le_40: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_163, 0);  alias_163 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, getitem_199);  le_40 = scalar_tensor_40 = getitem_199 = None
    unsqueeze_644: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_645: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_82: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_201: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_646)
    mul_647: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_40, sub_201);  sub_201 = None
    sum_83: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3]);  mul_647 = None
    mul_648: "f32[32]" = torch.ops.aten.mul.Tensor(sum_82, 9.964923469387754e-06)
    unsqueeze_647: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_648: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_649: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, 9.964923469387754e-06)
    mul_650: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_651: "f32[32]" = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
    unsqueeze_650: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_651: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_652: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_653: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_654: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    sub_202: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_646);  convolution = unsqueeze_646 = None
    mul_653: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_652);  sub_202 = unsqueeze_652 = None
    sub_203: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_40, mul_653);  where_40 = mul_653 = None
    sub_204: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_649);  sub_203 = unsqueeze_649 = None
    mul_654: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_655);  sub_204 = unsqueeze_655 = None
    mul_655: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_1);  sum_83 = squeeze_1 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_654, primals_249, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_654 = primals_249 = primals_1 = None
    getitem_203: "f32[32, 3, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[32]" = torch.ops.aten.copy_.default(primals_126, add_2);  primals_126 = add_2 = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_127, add_3);  primals_127 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_128, add);  primals_128 = add = None
    copy__3: "f32[64]" = torch.ops.aten.copy_.default(primals_129, add_7);  primals_129 = add_7 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_130, add_8);  primals_130 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_131, add_5);  primals_131 = add_5 = None
    copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_132, add_12);  primals_132 = add_12 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_133, add_13);  primals_133 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_134, add_10);  primals_134 = add_10 = None
    copy__9: "f32[32]" = torch.ops.aten.copy_.default(primals_135, add_17);  primals_135 = add_17 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_136, add_18);  primals_136 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_137, add_15);  primals_137 = add_15 = None
    copy__12: "f32[64]" = torch.ops.aten.copy_.default(primals_138, add_22);  primals_138 = add_22 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_139, add_23);  primals_139 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_140, add_20);  primals_140 = add_20 = None
    copy__15: "f32[32]" = torch.ops.aten.copy_.default(primals_141, add_27);  primals_141 = add_27 = None
    copy__16: "f32[32]" = torch.ops.aten.copy_.default(primals_142, add_28);  primals_142 = add_28 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_143, add_25);  primals_143 = add_25 = None
    copy__18: "f32[64]" = torch.ops.aten.copy_.default(primals_144, add_32);  primals_144 = add_32 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_145, add_33);  primals_145 = add_33 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_146, add_30);  primals_146 = add_30 = None
    copy__21: "f32[64]" = torch.ops.aten.copy_.default(primals_147, add_37);  primals_147 = add_37 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_148, add_38);  primals_148 = add_38 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_149, add_35);  primals_149 = add_35 = None
    copy__24: "f32[64]" = torch.ops.aten.copy_.default(primals_150, add_42);  primals_150 = add_42 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_151, add_43);  primals_151 = add_43 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_152, add_40);  primals_152 = add_40 = None
    copy__27: "f32[32]" = torch.ops.aten.copy_.default(primals_153, add_47);  primals_153 = add_47 = None
    copy__28: "f32[32]" = torch.ops.aten.copy_.default(primals_154, add_48);  primals_154 = add_48 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_155, add_45);  primals_155 = add_45 = None
    copy__30: "f32[64]" = torch.ops.aten.copy_.default(primals_156, add_52);  primals_156 = add_52 = None
    copy__31: "f32[64]" = torch.ops.aten.copy_.default(primals_157, add_53);  primals_157 = add_53 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_50);  primals_158 = add_50 = None
    copy__33: "f32[32]" = torch.ops.aten.copy_.default(primals_159, add_57);  primals_159 = add_57 = None
    copy__34: "f32[32]" = torch.ops.aten.copy_.default(primals_160, add_58);  primals_160 = add_58 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_161, add_55);  primals_161 = add_55 = None
    copy__36: "f32[128]" = torch.ops.aten.copy_.default(primals_162, add_62);  primals_162 = add_62 = None
    copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_163, add_63);  primals_163 = add_63 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_164, add_60);  primals_164 = add_60 = None
    copy__39: "f32[144]" = torch.ops.aten.copy_.default(primals_165, add_67);  primals_165 = add_67 = None
    copy__40: "f32[144]" = torch.ops.aten.copy_.default(primals_166, add_68);  primals_166 = add_68 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_65);  primals_167 = add_65 = None
    copy__42: "f32[144]" = torch.ops.aten.copy_.default(primals_168, add_72);  primals_168 = add_72 = None
    copy__43: "f32[144]" = torch.ops.aten.copy_.default(primals_169, add_73);  primals_169 = add_73 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_170, add_70);  primals_170 = add_70 = None
    copy__45: "f32[72]" = torch.ops.aten.copy_.default(primals_171, add_77);  primals_171 = add_77 = None
    copy__46: "f32[72]" = torch.ops.aten.copy_.default(primals_172, add_78);  primals_172 = add_78 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_75);  primals_173 = add_75 = None
    copy__48: "f32[144]" = torch.ops.aten.copy_.default(primals_174, add_82);  primals_174 = add_82 = None
    copy__49: "f32[144]" = torch.ops.aten.copy_.default(primals_175, add_83);  primals_175 = add_83 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_176, add_80);  primals_176 = add_80 = None
    copy__51: "f32[72]" = torch.ops.aten.copy_.default(primals_177, add_87);  primals_177 = add_87 = None
    copy__52: "f32[72]" = torch.ops.aten.copy_.default(primals_178, add_88);  primals_178 = add_88 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_85);  primals_179 = add_85 = None
    copy__54: "f32[144]" = torch.ops.aten.copy_.default(primals_180, add_92);  primals_180 = add_92 = None
    copy__55: "f32[144]" = torch.ops.aten.copy_.default(primals_181, add_93);  primals_181 = add_93 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_182, add_90);  primals_182 = add_90 = None
    copy__57: "f32[144]" = torch.ops.aten.copy_.default(primals_183, add_97);  primals_183 = add_97 = None
    copy__58: "f32[144]" = torch.ops.aten.copy_.default(primals_184, add_98);  primals_184 = add_98 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_95);  primals_185 = add_95 = None
    copy__60: "f32[144]" = torch.ops.aten.copy_.default(primals_186, add_102);  primals_186 = add_102 = None
    copy__61: "f32[144]" = torch.ops.aten.copy_.default(primals_187, add_103);  primals_187 = add_103 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_188, add_100);  primals_188 = add_100 = None
    copy__63: "f32[72]" = torch.ops.aten.copy_.default(primals_189, add_107);  primals_189 = add_107 = None
    copy__64: "f32[72]" = torch.ops.aten.copy_.default(primals_190, add_108);  primals_190 = add_108 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_105);  primals_191 = add_105 = None
    copy__66: "f32[144]" = torch.ops.aten.copy_.default(primals_192, add_112);  primals_192 = add_112 = None
    copy__67: "f32[144]" = torch.ops.aten.copy_.default(primals_193, add_113);  primals_193 = add_113 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_194, add_110);  primals_194 = add_110 = None
    copy__69: "f32[72]" = torch.ops.aten.copy_.default(primals_195, add_117);  primals_195 = add_117 = None
    copy__70: "f32[72]" = torch.ops.aten.copy_.default(primals_196, add_118);  primals_196 = add_118 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_115);  primals_197 = add_115 = None
    copy__72: "f32[288]" = torch.ops.aten.copy_.default(primals_198, add_122);  primals_198 = add_122 = None
    copy__73: "f32[288]" = torch.ops.aten.copy_.default(primals_199, add_123);  primals_199 = add_123 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_200, add_120);  primals_200 = add_120 = None
    copy__75: "f32[304]" = torch.ops.aten.copy_.default(primals_201, add_127);  primals_201 = add_127 = None
    copy__76: "f32[304]" = torch.ops.aten.copy_.default(primals_202, add_128);  primals_202 = add_128 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_125);  primals_203 = add_125 = None
    copy__78: "f32[304]" = torch.ops.aten.copy_.default(primals_204, add_132);  primals_204 = add_132 = None
    copy__79: "f32[304]" = torch.ops.aten.copy_.default(primals_205, add_133);  primals_205 = add_133 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_206, add_130);  primals_206 = add_130 = None
    copy__81: "f32[152]" = torch.ops.aten.copy_.default(primals_207, add_137);  primals_207 = add_137 = None
    copy__82: "f32[152]" = torch.ops.aten.copy_.default(primals_208, add_138);  primals_208 = add_138 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_135);  primals_209 = add_135 = None
    copy__84: "f32[304]" = torch.ops.aten.copy_.default(primals_210, add_142);  primals_210 = add_142 = None
    copy__85: "f32[304]" = torch.ops.aten.copy_.default(primals_211, add_143);  primals_211 = add_143 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_212, add_140);  primals_212 = add_140 = None
    copy__87: "f32[152]" = torch.ops.aten.copy_.default(primals_213, add_147);  primals_213 = add_147 = None
    copy__88: "f32[152]" = torch.ops.aten.copy_.default(primals_214, add_148);  primals_214 = add_148 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_145);  primals_215 = add_145 = None
    copy__90: "f32[304]" = torch.ops.aten.copy_.default(primals_216, add_152);  primals_216 = add_152 = None
    copy__91: "f32[304]" = torch.ops.aten.copy_.default(primals_217, add_153);  primals_217 = add_153 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_218, add_150);  primals_218 = add_150 = None
    copy__93: "f32[304]" = torch.ops.aten.copy_.default(primals_219, add_157);  primals_219 = add_157 = None
    copy__94: "f32[304]" = torch.ops.aten.copy_.default(primals_220, add_158);  primals_220 = add_158 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_221, add_155);  primals_221 = add_155 = None
    copy__96: "f32[304]" = torch.ops.aten.copy_.default(primals_222, add_162);  primals_222 = add_162 = None
    copy__97: "f32[304]" = torch.ops.aten.copy_.default(primals_223, add_163);  primals_223 = add_163 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_224, add_160);  primals_224 = add_160 = None
    copy__99: "f32[152]" = torch.ops.aten.copy_.default(primals_225, add_167);  primals_225 = add_167 = None
    copy__100: "f32[152]" = torch.ops.aten.copy_.default(primals_226, add_168);  primals_226 = add_168 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_227, add_165);  primals_227 = add_165 = None
    copy__102: "f32[304]" = torch.ops.aten.copy_.default(primals_228, add_172);  primals_228 = add_172 = None
    copy__103: "f32[304]" = torch.ops.aten.copy_.default(primals_229, add_173);  primals_229 = add_173 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_230, add_170);  primals_230 = add_170 = None
    copy__105: "f32[152]" = torch.ops.aten.copy_.default(primals_231, add_177);  primals_231 = add_177 = None
    copy__106: "f32[152]" = torch.ops.aten.copy_.default(primals_232, add_178);  primals_232 = add_178 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_233, add_175);  primals_233 = add_175 = None
    copy__108: "f32[480]" = torch.ops.aten.copy_.default(primals_234, add_182);  primals_234 = add_182 = None
    copy__109: "f32[480]" = torch.ops.aten.copy_.default(primals_235, add_183);  primals_235 = add_183 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_236, add_180);  primals_236 = add_180 = None
    copy__111: "f32[960]" = torch.ops.aten.copy_.default(primals_237, add_187);  primals_237 = add_187 = None
    copy__112: "f32[960]" = torch.ops.aten.copy_.default(primals_238, add_188);  primals_238 = add_188 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_185);  primals_239 = add_185 = None
    copy__114: "f32[1024]" = torch.ops.aten.copy_.default(primals_240, add_192);  primals_240 = add_192 = None
    copy__115: "f32[1024]" = torch.ops.aten.copy_.default(primals_241, add_193);  primals_241 = add_193 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_242, add_190);  primals_242 = add_190 = None
    copy__117: "f32[1280]" = torch.ops.aten.copy_.default(primals_243, add_197);  primals_243 = add_197 = None
    copy__118: "f32[1280]" = torch.ops.aten.copy_.default(primals_244, add_198);  primals_244 = add_198 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_195);  primals_245 = add_195 = None
    copy__120: "f32[1024]" = torch.ops.aten.copy_.default(primals_246, add_202);  primals_246 = add_202 = None
    copy__121: "f32[1024]" = torch.ops.aten.copy_.default(primals_247, add_203);  primals_247 = add_203 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_248, add_200);  primals_248 = add_200 = None
    return pytree.tree_unflatten([addmm, getitem_203, mul_655, sum_82, getitem_200, mul_646, sum_80, getitem_197, mul_637, sum_78, getitem_194, mul_628, sum_76, getitem_191, mul_619, sum_74, getitem_188, mul_610, sum_72, getitem_185, mul_601, sum_70, getitem_182, mul_592, sum_68, getitem_179, mul_583, sum_66, getitem_176, mul_574, sum_64, getitem_173, mul_565, sum_62, getitem_170, mul_556, sum_60, getitem_167, mul_547, sum_58, getitem_164, mul_538, sum_56, getitem_161, mul_529, sum_54, getitem_158, mul_520, sum_52, getitem_155, mul_511, sum_50, getitem_152, mul_502, sum_48, getitem_149, mul_493, sum_46, getitem_146, mul_484, sum_44, getitem_143, mul_475, sum_42, getitem_140, mul_466, sum_40, getitem_137, mul_457, sum_38, getitem_134, mul_448, sum_36, getitem_131, mul_439, sum_34, getitem_128, mul_430, sum_32, getitem_125, mul_421, sum_30, getitem_122, mul_412, sum_28, getitem_119, mul_403, sum_26, getitem_116, mul_394, sum_24, getitem_113, mul_385, sum_22, getitem_110, mul_376, sum_20, getitem_107, mul_367, sum_18, getitem_104, mul_358, sum_16, getitem_101, mul_349, sum_14, getitem_98, mul_340, sum_12, getitem_95, mul_331, sum_10, getitem_92, mul_322, sum_8, getitem_89, mul_313, sum_6, getitem_86, mul_304, sum_4, getitem_83, mul_295, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    