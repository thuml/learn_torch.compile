from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 14, 14, 384]"; primals_2: "f32[1, 1, 384]"; primals_3: "f32[64, 3, 7, 7]"; primals_4: "f32[64]"; primals_5: "f32[64]"; primals_6: "f32[64, 64, 3, 3]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64, 64, 3, 3]"; primals_10: "f32[64]"; primals_11: "f32[64]"; primals_12: "f32[192, 64, 4, 4]"; primals_13: "f32[192]"; primals_14: "f32[192]"; primals_15: "f32[192]"; primals_16: "f32[192, 192]"; primals_17: "f32[486, 192]"; primals_18: "f32[486]"; primals_19: "f32[192, 192]"; primals_20: "f32[192]"; primals_21: "f32[192]"; primals_22: "f32[192]"; primals_23: "f32[576, 192]"; primals_24: "f32[576]"; primals_25: "f32[192, 576]"; primals_26: "f32[192]"; primals_27: "f32[192]"; primals_28: "f32[192]"; primals_29: "f32[192, 192]"; primals_30: "f32[486, 192]"; primals_31: "f32[486]"; primals_32: "f32[192, 192]"; primals_33: "f32[192]"; primals_34: "f32[192]"; primals_35: "f32[192]"; primals_36: "f32[576, 192]"; primals_37: "f32[576]"; primals_38: "f32[192, 576]"; primals_39: "f32[192]"; primals_40: "f32[192]"; primals_41: "f32[192]"; primals_42: "f32[192, 192]"; primals_43: "f32[486, 192]"; primals_44: "f32[486]"; primals_45: "f32[192, 192]"; primals_46: "f32[192]"; primals_47: "f32[192]"; primals_48: "f32[192]"; primals_49: "f32[576, 192]"; primals_50: "f32[576]"; primals_51: "f32[192, 576]"; primals_52: "f32[192]"; primals_53: "f32[192]"; primals_54: "f32[192]"; primals_55: "f32[192, 192]"; primals_56: "f32[486, 192]"; primals_57: "f32[486]"; primals_58: "f32[192, 192]"; primals_59: "f32[192]"; primals_60: "f32[192]"; primals_61: "f32[192]"; primals_62: "f32[576, 192]"; primals_63: "f32[576]"; primals_64: "f32[192, 576]"; primals_65: "f32[192]"; primals_66: "f32[384, 192, 2, 2]"; primals_67: "f32[384]"; primals_68: "f32[384]"; primals_69: "f32[384]"; primals_70: "f32[1152, 384]"; primals_71: "f32[384, 384]"; primals_72: "f32[384]"; primals_73: "f32[384]"; primals_74: "f32[384]"; primals_75: "f32[1152, 384]"; primals_76: "f32[1152]"; primals_77: "f32[384, 1152]"; primals_78: "f32[384]"; primals_79: "f32[384]"; primals_80: "f32[384]"; primals_81: "f32[1152, 384]"; primals_82: "f32[384, 384]"; primals_83: "f32[384]"; primals_84: "f32[384]"; primals_85: "f32[384]"; primals_86: "f32[1152, 384]"; primals_87: "f32[1152]"; primals_88: "f32[384, 1152]"; primals_89: "f32[384]"; primals_90: "f32[384]"; primals_91: "f32[384]"; primals_92: "f32[1152, 384]"; primals_93: "f32[384, 384]"; primals_94: "f32[384]"; primals_95: "f32[384]"; primals_96: "f32[384]"; primals_97: "f32[1152, 384]"; primals_98: "f32[1152]"; primals_99: "f32[384, 1152]"; primals_100: "f32[384]"; primals_101: "f32[384]"; primals_102: "f32[384]"; primals_103: "f32[1152, 384]"; primals_104: "f32[384, 384]"; primals_105: "f32[384]"; primals_106: "f32[384]"; primals_107: "f32[384]"; primals_108: "f32[1152, 384]"; primals_109: "f32[1152]"; primals_110: "f32[384, 1152]"; primals_111: "f32[384]"; primals_112: "f32[384]"; primals_113: "f32[384]"; primals_114: "f32[1152, 384]"; primals_115: "f32[384, 384]"; primals_116: "f32[384]"; primals_117: "f32[384]"; primals_118: "f32[384]"; primals_119: "f32[1152, 384]"; primals_120: "f32[1152]"; primals_121: "f32[384, 1152]"; primals_122: "f32[384]"; primals_123: "f32[384]"; primals_124: "f32[384]"; primals_125: "f32[1152, 384]"; primals_126: "f32[384, 384]"; primals_127: "f32[384]"; primals_128: "f32[384]"; primals_129: "f32[384]"; primals_130: "f32[1152, 384]"; primals_131: "f32[1152]"; primals_132: "f32[384, 1152]"; primals_133: "f32[384]"; primals_134: "f32[384]"; primals_135: "f32[384]"; primals_136: "f32[1152, 384]"; primals_137: "f32[384, 384]"; primals_138: "f32[384]"; primals_139: "f32[384]"; primals_140: "f32[384]"; primals_141: "f32[1152, 384]"; primals_142: "f32[1152]"; primals_143: "f32[384, 1152]"; primals_144: "f32[384]"; primals_145: "f32[384]"; primals_146: "f32[384]"; primals_147: "f32[1152, 384]"; primals_148: "f32[384, 384]"; primals_149: "f32[384]"; primals_150: "f32[384]"; primals_151: "f32[384]"; primals_152: "f32[1152, 384]"; primals_153: "f32[1152]"; primals_154: "f32[384, 1152]"; primals_155: "f32[384]"; primals_156: "f32[384]"; primals_157: "f32[384]"; primals_158: "f32[1152, 384]"; primals_159: "f32[384, 384]"; primals_160: "f32[384]"; primals_161: "f32[384]"; primals_162: "f32[384]"; primals_163: "f32[1152, 384]"; primals_164: "f32[1152]"; primals_165: "f32[384, 1152]"; primals_166: "f32[384]"; primals_167: "f32[384]"; primals_168: "f32[384]"; primals_169: "f32[1152, 384]"; primals_170: "f32[384, 384]"; primals_171: "f32[384]"; primals_172: "f32[384]"; primals_173: "f32[384]"; primals_174: "f32[1152, 384]"; primals_175: "f32[1152]"; primals_176: "f32[384, 1152]"; primals_177: "f32[384]"; primals_178: "f32[384]"; primals_179: "f32[384]"; primals_180: "f32[1152, 384]"; primals_181: "f32[384, 384]"; primals_182: "f32[384]"; primals_183: "f32[384]"; primals_184: "f32[384]"; primals_185: "f32[1152, 384]"; primals_186: "f32[1152]"; primals_187: "f32[384, 1152]"; primals_188: "f32[384]"; primals_189: "f32[384]"; primals_190: "f32[384]"; primals_191: "f32[1152, 384]"; primals_192: "f32[384, 384]"; primals_193: "f32[384]"; primals_194: "f32[384]"; primals_195: "f32[384]"; primals_196: "f32[1152, 384]"; primals_197: "f32[1152]"; primals_198: "f32[384, 1152]"; primals_199: "f32[384]"; primals_200: "f32[384]"; primals_201: "f32[384]"; primals_202: "f32[1152, 384]"; primals_203: "f32[384, 384]"; primals_204: "f32[384]"; primals_205: "f32[384]"; primals_206: "f32[384]"; primals_207: "f32[1152, 384]"; primals_208: "f32[1152]"; primals_209: "f32[384, 1152]"; primals_210: "f32[384]"; primals_211: "f32[384]"; primals_212: "f32[384]"; primals_213: "f32[1152, 384]"; primals_214: "f32[384, 384]"; primals_215: "f32[384]"; primals_216: "f32[384]"; primals_217: "f32[384]"; primals_218: "f32[1152, 384]"; primals_219: "f32[1152]"; primals_220: "f32[384, 1152]"; primals_221: "f32[384]"; primals_222: "f32[384]"; primals_223: "f32[384]"; primals_224: "f32[768, 384]"; primals_225: "f32[384, 384]"; primals_226: "f32[384, 384]"; primals_227: "f32[384]"; primals_228: "f32[384]"; primals_229: "f32[384]"; primals_230: "f32[1152, 384]"; primals_231: "f32[1152]"; primals_232: "f32[384, 1152]"; primals_233: "f32[384]"; primals_234: "f32[384]"; primals_235: "f32[384]"; primals_236: "f32[768, 384]"; primals_237: "f32[384, 384]"; primals_238: "f32[384, 384]"; primals_239: "f32[384]"; primals_240: "f32[384]"; primals_241: "f32[384]"; primals_242: "f32[1152, 384]"; primals_243: "f32[1152]"; primals_244: "f32[384, 1152]"; primals_245: "f32[384]"; primals_246: "f32[384]"; primals_247: "f32[384]"; primals_248: "f32[1000, 384]"; primals_249: "f32[1000]"; primals_250: "f32[1000, 384]"; primals_251: "f32[1000]"; primals_252: "f32[64]"; primals_253: "f32[64]"; primals_254: "i64[]"; primals_255: "f32[64]"; primals_256: "f32[64]"; primals_257: "i64[]"; primals_258: "f32[64]"; primals_259: "f32[64]"; primals_260: "i64[]"; primals_261: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_261, primals_3, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)
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
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    convolution_1: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_6, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1);  primals_11 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    relu_2: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    convolution_3: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_2, primals_12, primals_13, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    permute: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 28, 28, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 28, 28, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_3: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone, getitem_7);  clone = None
    mul_21: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_22: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_21, primals_14);  mul_21 = None
    add_16: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_22, primals_15);  mul_22 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_1: "f32[192, 192]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    view: "f32[6272, 192]" = torch.ops.aten.view.default(add_16, [6272, 192])
    mm: "f32[6272, 192]" = torch.ops.aten.mm.default(view, permute_1)
    view_1: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm, [8, 28, 28, 192]);  mm = None
    permute_2: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_1, [0, 3, 1, 2]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_12: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_13: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_17: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_14: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    iota_3: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_15: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    add_18: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_14, unsqueeze_15);  unsqueeze_14 = unsqueeze_15 = None
    constant_pad_nd: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_2, [1, 1, 1, 1], 0.0);  permute_2 = None
    unsqueeze_16: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_17, -1);  add_17 = None
    unsqueeze_17: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    slice_1: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 0, 0, 9223372036854775807);  constant_pad_nd = None
    slice_2: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    index: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_2, [None, None, unsqueeze_17, add_18]);  slice_2 = unsqueeze_17 = add_18 = None
    permute_3: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    clone_1: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    view_2: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_1, [8, 1728, 196]);  clone_1 = None
    view_3: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_2, [8, 6, 32, 9, 196]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_3, [0, 1, 4, 3, 2]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_5: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
    avg_pool2d: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_5, [2, 2], [2, 2], [0, 0], True)
    permute_6: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d, [0, 2, 3, 1]);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_4: "f32[1568, 192]" = torch.ops.aten.view.default(permute_6, [1568, 192]);  permute_6 = None
    permute_7: "f32[192, 486]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_18, view_4, permute_7);  primals_18 = None
    view_5: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm, [8, 14, 14, 486]);  addmm = None
    view_6: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_5, [8, 196, 6, 9, 9]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_8: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_23: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_8, 0.1767766952966369);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_23, memory_format = torch.contiguous_format);  mul_23 = None
    amax: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_2, [-1], True)
    sub_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_2, amax);  clone_2 = amax = None
    exp: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_3, [8, 6, 196, 9, 9]);  clone_3 = None
    view_7: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand, [9408, 9, 9]);  expand = None
    expand_1: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_4, [8, 6, 196, 9, 32]);  permute_4 = None
    clone_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_8: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_4, [9408, 9, 32]);  clone_4 = None
    bmm: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm, [8, 6, 196, 9, 32]);  bmm = None
    permute_9: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_9, [0, 1, 4, 3, 2]);  view_9 = None
    clone_5: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_10: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_5, [8, 1728, 196]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_11: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_10, [8, 192, 3, 3, 14, 14]);  view_10 = None
    permute_10: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_11, [0, 1, 2, 4, 3, 5]);  view_11 = None
    iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_18: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    iota_5: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_19: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
    add_19: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_18, unsqueeze_19);  unsqueeze_18 = unsqueeze_19 = None
    unsqueeze_20: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_19, -1);  add_19 = None
    unsqueeze_21: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_22: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
    iota_7: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_23: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
    add_20: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_22, unsqueeze_23);  unsqueeze_22 = unsqueeze_23 = None
    full: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full, [None, None, unsqueeze_21, add_20], permute_10, True);  full = unsqueeze_21 = add_20 = permute_10 = None
    constant_pad_nd_1: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put, [-1, -1, -1, -1], 0.0);  _unsafe_index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_11: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_1, [0, 2, 3, 1]);  constant_pad_nd_1 = None
    permute_12: "f32[192, 192]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    clone_6: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_12: "f32[6272, 192]" = torch.ops.aten.view.default(clone_6, [6272, 192]);  clone_6 = None
    mm_1: "f32[6272, 192]" = torch.ops.aten.mm.default(view_12, permute_12)
    view_13: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_1, [8, 28, 28, 192]);  mm_1 = None
    add_21: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_13, primals_20);  view_13 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_7: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_22: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_8: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_8, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 28, 28, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 28, 28, 1]" = var_mean_4[1];  var_mean_4 = None
    add_23: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_5: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_8, getitem_9);  clone_8 = None
    mul_24: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
    mul_25: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_24, primals_21);  mul_24 = None
    add_24: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_25, primals_22);  mul_25 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[6272, 192]" = torch.ops.aten.view.default(add_24, [6272, 192]);  add_24 = None
    permute_13: "f32[192, 576]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_1: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_24, view_14, permute_13);  primals_24 = None
    view_15: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_1, [8, 28, 28, 576]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_27: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_25: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_28: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_26, add_25);  mul_26 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[6272, 576]" = torch.ops.aten.view.default(clone_9, [6272, 576]);  clone_9 = None
    permute_14: "f32[576, 192]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_2: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_26, view_16, permute_14);  primals_26 = None
    view_17: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_2, [8, 28, 28, 192]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_26: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_22, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_11: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_11, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 28, 28, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_6: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_11, getitem_11);  clone_11 = None
    mul_29: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    mul_30: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_29, primals_27);  mul_29 = None
    add_28: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_30, primals_28);  mul_30 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_15: "f32[192, 192]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    view_18: "f32[6272, 192]" = torch.ops.aten.view.default(add_28, [6272, 192])
    mm_2: "f32[6272, 192]" = torch.ops.aten.mm.default(view_18, permute_15)
    view_19: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_2, [8, 28, 28, 192]);  mm_2 = None
    permute_16: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_19, [0, 3, 1, 2]);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_24: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
    iota_9: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_25: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
    add_29: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
    iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_26: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = None
    iota_11: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_27: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
    add_30: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
    constant_pad_nd_2: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_16, [1, 1, 1, 1], 0.0);  permute_16 = None
    unsqueeze_28: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_29, -1);  add_29 = None
    unsqueeze_29: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    slice_3: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, 0, 0, 9223372036854775807);  constant_pad_nd_2 = None
    slice_4: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    index_1: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_4, [None, None, unsqueeze_29, add_30]);  slice_4 = unsqueeze_29 = add_30 = None
    permute_17: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    clone_12: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_20: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_12, [8, 1728, 196]);  clone_12 = None
    view_21: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_20, [8, 6, 32, 9, 196]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_18: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_21, [0, 1, 4, 3, 2]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_19: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_28, [0, 3, 1, 2]);  add_28 = None
    avg_pool2d_1: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_19, [2, 2], [2, 2], [0, 0], True)
    permute_20: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_1, [0, 2, 3, 1]);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_22: "f32[1568, 192]" = torch.ops.aten.view.default(permute_20, [1568, 192]);  permute_20 = None
    permute_21: "f32[192, 486]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_3: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_31, view_22, permute_21);  primals_31 = None
    view_23: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_3, [8, 14, 14, 486]);  addmm_3 = None
    view_24: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_23, [8, 196, 6, 9, 9]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_22: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_31: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_22, 0.1767766952966369);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_31, memory_format = torch.contiguous_format);  mul_31 = None
    amax_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_13, [-1], True)
    sub_7: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_13, amax_1);  clone_13 = amax_1 = None
    exp_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_14: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_14, [8, 6, 196, 9, 9]);  clone_14 = None
    view_25: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_2, [9408, 9, 9]);  expand_2 = None
    expand_3: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_18, [8, 6, 196, 9, 32]);  permute_18 = None
    clone_15: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_26: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_15, [9408, 9, 32]);  clone_15 = None
    bmm_1: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_1, [8, 6, 196, 9, 32]);  bmm_1 = None
    permute_23: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_27, [0, 1, 4, 3, 2]);  view_27 = None
    clone_16: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_28: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_16, [8, 1728, 196]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_29: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_28, [8, 192, 3, 3, 14, 14]);  view_28 = None
    permute_24: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_29, [0, 1, 2, 4, 3, 5]);  view_29 = None
    iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_30: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
    iota_13: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_31: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
    add_31: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
    unsqueeze_32: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_31, -1);  add_31 = None
    unsqueeze_33: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_34: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = None
    iota_15: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_35: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
    add_32: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_34, unsqueeze_35);  unsqueeze_34 = unsqueeze_35 = None
    full_1: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_1, [None, None, unsqueeze_33, add_32], permute_24, True);  full_1 = unsqueeze_33 = add_32 = permute_24 = None
    constant_pad_nd_3: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_25: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_3, [0, 2, 3, 1]);  constant_pad_nd_3 = None
    permute_26: "f32[192, 192]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    clone_17: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_30: "f32[6272, 192]" = torch.ops.aten.view.default(clone_17, [6272, 192]);  clone_17 = None
    mm_3: "f32[6272, 192]" = torch.ops.aten.mm.default(view_30, permute_26)
    view_31: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_3, [8, 28, 28, 192]);  mm_3 = None
    add_33: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_31, primals_33);  view_31 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_18: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_34: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_26, clone_18);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_19: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_19, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 28, 28, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
    add_35: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_8: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_19, getitem_13);  clone_19 = None
    mul_32: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    mul_33: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_32, primals_34);  mul_32 = None
    add_36: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_33, primals_35);  mul_33 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[6272, 192]" = torch.ops.aten.view.default(add_36, [6272, 192]);  add_36 = None
    permute_27: "f32[192, 576]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_4: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_37, view_32, permute_27);  primals_37 = None
    view_33: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_4, [8, 28, 28, 576]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_34: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_35: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_1: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_37: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_36: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_34, add_37);  mul_34 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[6272, 576]" = torch.ops.aten.view.default(clone_20, [6272, 576]);  clone_20 = None
    permute_28: "f32[576, 192]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_5: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_39, view_34, permute_28);  primals_39 = None
    view_35: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_5, [8, 28, 28, 192]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_35);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_38: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_34, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_22: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_22, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 28, 28, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 28, 28, 1]" = var_mean_7[1];  var_mean_7 = None
    add_39: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_9: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_22, getitem_15);  clone_22 = None
    mul_37: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
    mul_38: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_37, primals_40);  mul_37 = None
    add_40: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_38, primals_41);  mul_38 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_29: "f32[192, 192]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_36: "f32[6272, 192]" = torch.ops.aten.view.default(add_40, [6272, 192])
    mm_4: "f32[6272, 192]" = torch.ops.aten.mm.default(view_36, permute_29)
    view_37: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_4, [8, 28, 28, 192]);  mm_4 = None
    permute_30: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_37, [0, 3, 1, 2]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_36: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
    iota_17: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_37: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
    add_41: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_36, unsqueeze_37);  unsqueeze_36 = unsqueeze_37 = None
    iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_38: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = None
    iota_19: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_39: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
    add_42: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
    constant_pad_nd_4: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_30, [1, 1, 1, 1], 0.0);  permute_30 = None
    unsqueeze_40: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_41, -1);  add_41 = None
    unsqueeze_41: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    slice_5: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, 0, 0, 9223372036854775807);  constant_pad_nd_4 = None
    slice_6: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    index_2: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_6, [None, None, unsqueeze_41, add_42]);  slice_6 = unsqueeze_41 = add_42 = None
    permute_31: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    clone_23: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_38: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_23, [8, 1728, 196]);  clone_23 = None
    view_39: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_38, [8, 6, 32, 9, 196]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_32: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_39, [0, 1, 4, 3, 2]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_33: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
    avg_pool2d_2: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_33, [2, 2], [2, 2], [0, 0], True)
    permute_34: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_2, [0, 2, 3, 1]);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_40: "f32[1568, 192]" = torch.ops.aten.view.default(permute_34, [1568, 192]);  permute_34 = None
    permute_35: "f32[192, 486]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_6: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_44, view_40, permute_35);  primals_44 = None
    view_41: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_6, [8, 14, 14, 486]);  addmm_6 = None
    view_42: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_41, [8, 196, 6, 9, 9]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_36: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3, 4]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_39: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_36, 0.1767766952966369);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_24: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_39, memory_format = torch.contiguous_format);  mul_39 = None
    amax_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_24, [-1], True)
    sub_10: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_24, amax_2);  clone_24 = amax_2 = None
    exp_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_5: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_25: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_25, [8, 6, 196, 9, 9]);  clone_25 = None
    view_43: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_4, [9408, 9, 9]);  expand_4 = None
    expand_5: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_32, [8, 6, 196, 9, 32]);  permute_32 = None
    clone_26: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_44: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_26, [9408, 9, 32]);  clone_26 = None
    bmm_2: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_2, [8, 6, 196, 9, 32]);  bmm_2 = None
    permute_37: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_45, [0, 1, 4, 3, 2]);  view_45 = None
    clone_27: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_46: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_27, [8, 1728, 196]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_47: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_46, [8, 192, 3, 3, 14, 14]);  view_46 = None
    permute_38: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_47, [0, 1, 2, 4, 3, 5]);  view_47 = None
    iota_20: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_42: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
    iota_21: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_43: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
    add_43: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_42, unsqueeze_43);  unsqueeze_42 = unsqueeze_43 = None
    unsqueeze_44: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_43, -1);  add_43 = None
    unsqueeze_45: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    iota_22: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_46: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = None
    iota_23: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_47: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
    add_44: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_46, unsqueeze_47);  unsqueeze_46 = unsqueeze_47 = None
    full_2: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_2, [None, None, unsqueeze_45, add_44], permute_38, True);  full_2 = unsqueeze_45 = add_44 = permute_38 = None
    constant_pad_nd_5: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_2, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_39: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_5, [0, 2, 3, 1]);  constant_pad_nd_5 = None
    permute_40: "f32[192, 192]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_28: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_48: "f32[6272, 192]" = torch.ops.aten.view.default(clone_28, [6272, 192]);  clone_28 = None
    mm_5: "f32[6272, 192]" = torch.ops.aten.mm.default(view_48, permute_40)
    view_49: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_5, [8, 28, 28, 192]);  mm_5 = None
    add_45: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_49, primals_46);  view_49 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_29: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_38, clone_29);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_30: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_30, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 28, 28, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
    add_47: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_11: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_30, getitem_17);  clone_30 = None
    mul_40: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
    mul_41: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_40, primals_47);  mul_40 = None
    add_48: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_41, primals_48);  mul_41 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 192]" = torch.ops.aten.view.default(add_48, [6272, 192]);  add_48 = None
    permute_41: "f32[192, 576]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_7: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_50, view_50, permute_41);  primals_50 = None
    view_51: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_7, [8, 28, 28, 576]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_42: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_43: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_49: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_44: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_42, add_49);  mul_42 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 576]" = torch.ops.aten.view.default(clone_31, [6272, 576]);  clone_31 = None
    permute_42: "f32[576, 192]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_8: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_52, view_52, permute_42);  primals_52 = None
    view_53: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_8, [8, 28, 28, 192]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_46, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_33: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_33, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 28, 28, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 28, 28, 1]" = var_mean_9[1];  var_mean_9 = None
    add_51: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_12: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_33, getitem_19);  clone_33 = None
    mul_45: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = None
    mul_46: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_45, primals_53);  mul_45 = None
    add_52: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_46, primals_54);  mul_46 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_43: "f32[192, 192]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_54: "f32[6272, 192]" = torch.ops.aten.view.default(add_52, [6272, 192])
    mm_6: "f32[6272, 192]" = torch.ops.aten.mm.default(view_54, permute_43)
    view_55: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_6, [8, 28, 28, 192]);  mm_6 = None
    permute_44: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_55, [0, 3, 1, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    iota_24: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_48: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
    iota_25: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_49: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
    add_53: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_48, unsqueeze_49);  unsqueeze_48 = unsqueeze_49 = None
    iota_26: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_50: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = None
    iota_27: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_51: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
    add_54: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_50, unsqueeze_51);  unsqueeze_50 = unsqueeze_51 = None
    constant_pad_nd_6: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_44, [1, 1, 1, 1], 0.0);  permute_44 = None
    unsqueeze_52: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_53, -1);  add_53 = None
    unsqueeze_53: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    slice_7: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, 0, 0, 9223372036854775807);  constant_pad_nd_6 = None
    slice_8: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    index_3: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_8, [None, None, unsqueeze_53, add_54]);  slice_8 = unsqueeze_53 = add_54 = None
    permute_45: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    clone_34: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_56: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_34, [8, 1728, 196]);  clone_34 = None
    view_57: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_56, [8, 6, 32, 9, 196]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_46: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_57, [0, 1, 4, 3, 2]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_47: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_52, [0, 3, 1, 2]);  add_52 = None
    avg_pool2d_3: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_47, [2, 2], [2, 2], [0, 0], True)
    permute_48: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_3, [0, 2, 3, 1]);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_58: "f32[1568, 192]" = torch.ops.aten.view.default(permute_48, [1568, 192]);  permute_48 = None
    permute_49: "f32[192, 486]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_9: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_57, view_58, permute_49);  primals_57 = None
    view_59: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_9, [8, 14, 14, 486]);  addmm_9 = None
    view_60: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_59, [8, 196, 6, 9, 9]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_50: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3, 4]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_47: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_50, 0.1767766952966369);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_35: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_47, memory_format = torch.contiguous_format);  mul_47 = None
    amax_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
    sub_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_35, amax_3);  clone_35 = amax_3 = None
    exp_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_4: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_36: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_36, [8, 6, 196, 9, 9]);  clone_36 = None
    view_61: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_6, [9408, 9, 9]);  expand_6 = None
    expand_7: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_46, [8, 6, 196, 9, 32]);  permute_46 = None
    clone_37: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_62: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_37, [9408, 9, 32]);  clone_37 = None
    bmm_3: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_3, [8, 6, 196, 9, 32]);  bmm_3 = None
    permute_51: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_63, [0, 1, 4, 3, 2]);  view_63 = None
    clone_38: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_64: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_38, [8, 1728, 196]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_65: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_64, [8, 192, 3, 3, 14, 14]);  view_64 = None
    permute_52: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_65, [0, 1, 2, 4, 3, 5]);  view_65 = None
    iota_28: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_54: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
    iota_29: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_55: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
    add_55: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_54, unsqueeze_55);  unsqueeze_54 = unsqueeze_55 = None
    unsqueeze_56: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_55, -1);  add_55 = None
    unsqueeze_57: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    iota_30: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_58: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = None
    iota_31: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_59: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
    add_56: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_58, unsqueeze_59);  unsqueeze_58 = unsqueeze_59 = None
    full_3: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_3: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_3, [None, None, unsqueeze_57, add_56], permute_52, True);  full_3 = unsqueeze_57 = add_56 = permute_52 = None
    constant_pad_nd_7: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_53: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_7, [0, 2, 3, 1]);  constant_pad_nd_7 = None
    permute_54: "f32[192, 192]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    clone_39: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_66: "f32[6272, 192]" = torch.ops.aten.view.default(clone_39, [6272, 192]);  clone_39 = None
    mm_7: "f32[6272, 192]" = torch.ops.aten.mm.default(view_66, permute_54)
    view_67: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_7, [8, 28, 28, 192]);  mm_7 = None
    add_57: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_67, primals_59);  view_67 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_40: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_58: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_50, clone_40);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_41: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_41, [3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 28, 28, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 28, 28, 1]" = var_mean_10[1];  var_mean_10 = None
    add_59: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_14: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_41, getitem_21);  clone_41 = None
    mul_48: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_49: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_48, primals_60);  mul_48 = None
    add_60: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_49, primals_61);  mul_49 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[6272, 192]" = torch.ops.aten.view.default(add_60, [6272, 192]);  add_60 = None
    permute_55: "f32[192, 576]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_10: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_63, view_68, permute_55);  primals_63 = None
    view_69: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_10, [8, 28, 28, 576]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_51: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_3: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_61: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_52: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_50, add_61);  mul_50 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_42: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_70: "f32[6272, 576]" = torch.ops.aten.view.default(clone_42, [6272, 576]);  clone_42 = None
    permute_56: "f32[576, 192]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_11: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_65, view_70, permute_56);  primals_65 = None
    view_71: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_11, [8, 28, 28, 192]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_43: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_62: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_58, clone_43);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_62, [0, 3, 1, 2]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    convolution_4: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(permute_57, primals_66, primals_67, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 384]" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    add_63: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(permute_58, primals_1);  permute_58 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:621, code: x = self.pos_drop(x)
    clone_44: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_45: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(clone_44, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_45, [3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 14, 14, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
    add_64: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_15: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_45, getitem_23);  clone_45 = None
    mul_53: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    mul_54: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_53, primals_68);  mul_53 = None
    add_65: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_54, primals_69);  mul_54 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_59: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_72: "f32[1568, 384]" = torch.ops.aten.view.default(add_65, [1568, 384]);  add_65 = None
    mm_8: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_72, permute_59)
    view_73: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_8, [8, 14, 14, 1152]);  mm_8 = None
    view_74: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_73, [8, 196, 3, 12, 32]);  view_73 = None
    permute_60: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_74, [2, 0, 3, 1, 4]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_60);  permute_60 = None
    getitem_24: "f32[8, 12, 196, 32]" = unbind[0]
    getitem_25: "f32[8, 12, 196, 32]" = unbind[1]
    getitem_26: "f32[8, 12, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_61: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
    expand_8: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_24, [8, 12, 196, 32]);  getitem_24 = None
    clone_46: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_75: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_46, [96, 196, 32]);  clone_46 = None
    expand_9: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_61, [8, 12, 32, 196]);  permute_61 = None
    clone_47: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_76: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_47, [96, 32, 196]);  clone_47 = None
    bmm_4: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 12, 196, 196]);  bmm_4 = None
    mul_55: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_77, 0.1767766952966369);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_55, [-1], True)
    sub_16: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_55, amax_4);  mul_55 = amax_4 = None
    exp_4: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_7: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_48: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_10: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_48, [8, 12, 196, 196]);  clone_48 = None
    view_78: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_10, [96, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_26, [8, 12, 196, 32]);  getitem_26 = None
    clone_49: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_79: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_49, [96, 196, 32]);  clone_49 = None
    bmm_5: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 12, 196, 32]);  bmm_5 = None
    permute_62: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_50: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_81: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_50, [8, 14, 14, 384]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_82: "f32[1568, 384]" = torch.ops.aten.view.default(view_81, [1568, 384]);  view_81 = None
    permute_63: "f32[384, 384]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_12: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_72, view_82, permute_63);  primals_72 = None
    view_83: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_12, [8, 14, 14, 384]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_51: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(clone_44, clone_51);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_52: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_52, [3], correction = 0, keepdim = True)
    getitem_27: "f32[8, 14, 14, 1]" = var_mean_12[0]
    getitem_28: "f32[8, 14, 14, 1]" = var_mean_12[1];  var_mean_12 = None
    add_67: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_27, 1e-05);  getitem_27 = None
    rsqrt_12: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_17: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_52, getitem_28);  clone_52 = None
    mul_56: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_57: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_56, primals_73);  mul_56 = None
    add_68: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_57, primals_74);  mul_57 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[1568, 384]" = torch.ops.aten.view.default(add_68, [1568, 384]);  add_68 = None
    permute_64: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_13: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_76, view_84, permute_64);  primals_76 = None
    view_85: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_13, [8, 14, 14, 1152]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_58: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_59: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_4: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_69: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_60: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_58, add_69);  mul_58 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_53, [1568, 1152]);  clone_53 = None
    permute_65: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_14: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_78, view_86, permute_65);  primals_78 = None
    view_87: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_14, [8, 14, 14, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_66, clone_54);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_55: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_55, [3], correction = 0, keepdim = True)
    getitem_29: "f32[8, 14, 14, 1]" = var_mean_13[0]
    getitem_30: "f32[8, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
    add_71: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_29, 1e-05);  getitem_29 = None
    rsqrt_13: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_18: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_55, getitem_30);  clone_55 = None
    mul_61: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
    mul_62: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_61, primals_79);  mul_61 = None
    add_72: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_62, primals_80);  mul_62 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_66: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_88: "f32[1568, 384]" = torch.ops.aten.view.default(add_72, [1568, 384]);  add_72 = None
    mm_9: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_88, permute_66)
    view_89: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_9, [8, 14, 14, 1152]);  mm_9 = None
    view_90: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_89, [8, 196, 3, 12, 32]);  view_89 = None
    permute_67: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 1, 4]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_31: "f32[8, 12, 196, 32]" = unbind_1[0]
    getitem_32: "f32[8, 12, 196, 32]" = unbind_1[1]
    getitem_33: "f32[8, 12, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_68: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_32, [0, 1, 3, 2]);  getitem_32 = None
    expand_12: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_31, [8, 12, 196, 32]);  getitem_31 = None
    clone_56: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_91: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_56, [96, 196, 32]);  clone_56 = None
    expand_13: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_68, [8, 12, 32, 196]);  permute_68 = None
    clone_57: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_92: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_57, [96, 32, 196]);  clone_57 = None
    bmm_6: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_91, view_92)
    view_93: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 12, 196, 196]);  bmm_6 = None
    mul_63: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_93, 0.1767766952966369);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_63, [-1], True)
    sub_19: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_63, amax_5);  mul_63 = amax_5 = None
    exp_5: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_8: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_58: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_14: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_58, [8, 12, 196, 196]);  clone_58 = None
    view_94: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_14, [96, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_33, [8, 12, 196, 32]);  getitem_33 = None
    clone_59: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_95: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_59, [96, 196, 32]);  clone_59 = None
    bmm_7: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_94, view_95)
    view_96: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 12, 196, 32]);  bmm_7 = None
    permute_69: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    clone_60: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_97: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_60, [8, 14, 14, 384]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_98: "f32[1568, 384]" = torch.ops.aten.view.default(view_97, [1568, 384]);  view_97 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_15: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_83, view_98, permute_70);  primals_83 = None
    view_99: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_15, [8, 14, 14, 384]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_61: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_73: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_70, clone_61);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_62: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_62, [3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 14, 14, 1]" = var_mean_14[0]
    getitem_35: "f32[8, 14, 14, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_14: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_20: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_62, getitem_35);  clone_62 = None
    mul_64: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_65: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_64, primals_84);  mul_64 = None
    add_75: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_65, primals_85);  mul_65 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[1568, 384]" = torch.ops.aten.view.default(add_75, [1568, 384]);  add_75 = None
    permute_71: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_16: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_87, view_100, permute_71);  primals_87 = None
    view_101: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_16, [8, 14, 14, 1152]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_66: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_67: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_5: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_76: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_68: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_66, add_76);  mul_66 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_63, [1568, 1152]);  clone_63 = None
    permute_72: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_89, view_102, permute_72);  primals_89 = None
    view_103: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_17, [8, 14, 14, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_77: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_73, clone_64);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_65: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_65, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 14, 14, 1]" = var_mean_15[0]
    getitem_37: "f32[8, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_15: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_21: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_65, getitem_37);  clone_65 = None
    mul_69: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
    mul_70: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_69, primals_90);  mul_69 = None
    add_79: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_70, primals_91);  mul_70 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_73: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_104: "f32[1568, 384]" = torch.ops.aten.view.default(add_79, [1568, 384]);  add_79 = None
    mm_10: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_104, permute_73)
    view_105: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_10, [8, 14, 14, 1152]);  mm_10 = None
    view_106: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_105, [8, 196, 3, 12, 32]);  view_105 = None
    permute_74: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_106, [2, 0, 3, 1, 4]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_38: "f32[8, 12, 196, 32]" = unbind_2[0]
    getitem_39: "f32[8, 12, 196, 32]" = unbind_2[1]
    getitem_40: "f32[8, 12, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_75: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_39, [0, 1, 3, 2]);  getitem_39 = None
    expand_16: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_38, [8, 12, 196, 32]);  getitem_38 = None
    clone_66: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_107: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_66, [96, 196, 32]);  clone_66 = None
    expand_17: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_75, [8, 12, 32, 196]);  permute_75 = None
    clone_67: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_108: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_67, [96, 32, 196]);  clone_67 = None
    bmm_8: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_107, view_108)
    view_109: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 12, 196, 196]);  bmm_8 = None
    mul_71: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_109, 0.1767766952966369);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_71, [-1], True)
    sub_22: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_71, amax_6);  mul_71 = amax_6 = None
    exp_6: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_7: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_9: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_68: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_18: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_68, [8, 12, 196, 196]);  clone_68 = None
    view_110: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_18, [96, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_40, [8, 12, 196, 32]);  getitem_40 = None
    clone_69: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_111: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_69, [96, 196, 32]);  clone_69 = None
    bmm_9: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_110, view_111)
    view_112: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_9, [8, 12, 196, 32]);  bmm_9 = None
    permute_76: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    clone_70: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_113: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_70, [8, 14, 14, 384]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_114: "f32[1568, 384]" = torch.ops.aten.view.default(view_113, [1568, 384]);  view_113 = None
    permute_77: "f32[384, 384]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_18: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_94, view_114, permute_77);  primals_94 = None
    view_115: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_18, [8, 14, 14, 384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_71: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_80: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_77, clone_71);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_72: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_72, [3], correction = 0, keepdim = True)
    getitem_41: "f32[8, 14, 14, 1]" = var_mean_16[0]
    getitem_42: "f32[8, 14, 14, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_16: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_72, getitem_42);  clone_72 = None
    mul_72: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
    mul_73: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_72, primals_95);  mul_72 = None
    add_82: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_73, primals_96);  mul_73 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1568, 384]" = torch.ops.aten.view.default(add_82, [1568, 384]);  add_82 = None
    permute_78: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_19: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_98, view_116, permute_78);  primals_98 = None
    view_117: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_19, [8, 14, 14, 1152]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_75: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_6: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_76: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_74, add_83);  mul_74 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_73: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_73, [1568, 1152]);  clone_73 = None
    permute_79: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_20: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_100, view_118, permute_79);  primals_100 = None
    view_119: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_20, [8, 14, 14, 384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_74: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_119);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_84: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_80, clone_74);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_75: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_84, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_75, [3], correction = 0, keepdim = True)
    getitem_43: "f32[8, 14, 14, 1]" = var_mean_17[0]
    getitem_44: "f32[8, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
    add_85: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05);  getitem_43 = None
    rsqrt_17: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_24: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_75, getitem_44);  clone_75 = None
    mul_77: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
    mul_78: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_77, primals_101);  mul_77 = None
    add_86: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_78, primals_102);  mul_78 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_80: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    view_120: "f32[1568, 384]" = torch.ops.aten.view.default(add_86, [1568, 384]);  add_86 = None
    mm_11: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_120, permute_80)
    view_121: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_11, [8, 14, 14, 1152]);  mm_11 = None
    view_122: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_121, [8, 196, 3, 12, 32]);  view_121 = None
    permute_81: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_81);  permute_81 = None
    getitem_45: "f32[8, 12, 196, 32]" = unbind_3[0]
    getitem_46: "f32[8, 12, 196, 32]" = unbind_3[1]
    getitem_47: "f32[8, 12, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 3, 2]);  getitem_46 = None
    expand_20: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_45, [8, 12, 196, 32]);  getitem_45 = None
    clone_76: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_123: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_76, [96, 196, 32]);  clone_76 = None
    expand_21: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_82, [8, 12, 32, 196]);  permute_82 = None
    clone_77: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_124: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_77, [96, 32, 196]);  clone_77 = None
    bmm_10: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_123, view_124)
    view_125: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 12, 196, 196]);  bmm_10 = None
    mul_79: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_125, 0.1767766952966369);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_79, [-1], True)
    sub_25: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_79, amax_7);  mul_79 = amax_7 = None
    exp_7: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_8: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_10: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_78: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_22: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_78, [8, 12, 196, 196]);  clone_78 = None
    view_126: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_22, [96, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_47, [8, 12, 196, 32]);  getitem_47 = None
    clone_79: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_127: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_79, [96, 196, 32]);  clone_79 = None
    bmm_11: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_126, view_127)
    view_128: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_11, [8, 12, 196, 32]);  bmm_11 = None
    permute_83: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_80: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_129: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_80, [8, 14, 14, 384]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_130: "f32[1568, 384]" = torch.ops.aten.view.default(view_129, [1568, 384]);  view_129 = None
    permute_84: "f32[384, 384]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_21: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_105, view_130, permute_84);  primals_105 = None
    view_131: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_21, [8, 14, 14, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_81: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_131);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_87: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_84, clone_81);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_82: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_82, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 14, 14, 1]" = var_mean_18[0]
    getitem_49: "f32[8, 14, 14, 1]" = var_mean_18[1];  var_mean_18 = None
    add_88: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_18: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_26: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_82, getitem_49);  clone_82 = None
    mul_80: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
    mul_81: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_80, primals_106);  mul_80 = None
    add_89: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_81, primals_107);  mul_81 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_132: "f32[1568, 384]" = torch.ops.aten.view.default(add_89, [1568, 384]);  add_89 = None
    permute_85: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_22: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_109, view_132, permute_85);  primals_109 = None
    view_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_22, [8, 14, 14, 1152]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
    mul_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
    erf_7: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_90: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_84: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_82, add_90);  mul_82 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_134: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_83, [1568, 1152]);  clone_83 = None
    permute_86: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_111, view_134, permute_86);  primals_111 = None
    view_135: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_23, [8, 14, 14, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_135);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_91: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_87, clone_84);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_85: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_91, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_85, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 14, 14, 1]" = var_mean_19[0]
    getitem_51: "f32[8, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
    add_92: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_19: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_27: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_85, getitem_51);  clone_85 = None
    mul_85: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
    mul_86: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_85, primals_112);  mul_85 = None
    add_93: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_86, primals_113);  mul_86 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_87: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_136: "f32[1568, 384]" = torch.ops.aten.view.default(add_93, [1568, 384]);  add_93 = None
    mm_12: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_136, permute_87)
    view_137: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_12, [8, 14, 14, 1152]);  mm_12 = None
    view_138: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_137, [8, 196, 3, 12, 32]);  view_137 = None
    permute_88: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_138, [2, 0, 3, 1, 4]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_88);  permute_88 = None
    getitem_52: "f32[8, 12, 196, 32]" = unbind_4[0]
    getitem_53: "f32[8, 12, 196, 32]" = unbind_4[1]
    getitem_54: "f32[8, 12, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_89: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 3, 2]);  getitem_53 = None
    expand_24: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_52, [8, 12, 196, 32]);  getitem_52 = None
    clone_86: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_139: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_86, [96, 196, 32]);  clone_86 = None
    expand_25: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_89, [8, 12, 32, 196]);  permute_89 = None
    clone_87: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_140: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_87, [96, 32, 196]);  clone_87 = None
    bmm_12: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_139, view_140)
    view_141: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 12, 196, 196]);  bmm_12 = None
    mul_87: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_141, 0.1767766952966369);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_87, [-1], True)
    sub_28: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_87, amax_8);  mul_87 = amax_8 = None
    exp_8: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_9: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_11: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_88: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_26: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_88, [8, 12, 196, 196]);  clone_88 = None
    view_142: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_26, [96, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_54, [8, 12, 196, 32]);  getitem_54 = None
    clone_89: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_143: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_89, [96, 196, 32]);  clone_89 = None
    bmm_13: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_142, view_143)
    view_144: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_13, [8, 12, 196, 32]);  bmm_13 = None
    permute_90: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_90: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_145: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_90, [8, 14, 14, 384]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_146: "f32[1568, 384]" = torch.ops.aten.view.default(view_145, [1568, 384]);  view_145 = None
    permute_91: "f32[384, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_24: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_116, view_146, permute_91);  primals_116 = None
    view_147: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_24, [8, 14, 14, 384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_91: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_147);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_94: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_91, clone_91);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_92: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_92, [3], correction = 0, keepdim = True)
    getitem_55: "f32[8, 14, 14, 1]" = var_mean_20[0]
    getitem_56: "f32[8, 14, 14, 1]" = var_mean_20[1];  var_mean_20 = None
    add_95: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
    rsqrt_20: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_29: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_92, getitem_56);  clone_92 = None
    mul_88: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
    mul_89: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_88, primals_117);  mul_88 = None
    add_96: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_89, primals_118);  mul_89 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_148: "f32[1568, 384]" = torch.ops.aten.view.default(add_96, [1568, 384]);  add_96 = None
    permute_92: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_25: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_120, view_148, permute_92);  primals_120 = None
    view_149: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_25, [8, 14, 14, 1152]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_90: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_91: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476)
    erf_8: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_97: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_92: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_90, add_97);  mul_90 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_93: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_150: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_93, [1568, 1152]);  clone_93 = None
    permute_93: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_26: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_122, view_150, permute_93);  primals_122 = None
    view_151: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_26, [8, 14, 14, 384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_94: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_98: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_94, clone_94);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_95: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_95, [3], correction = 0, keepdim = True)
    getitem_57: "f32[8, 14, 14, 1]" = var_mean_21[0]
    getitem_58: "f32[8, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
    add_99: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05);  getitem_57 = None
    rsqrt_21: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_30: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_58);  clone_95 = None
    mul_93: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
    mul_94: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_93, primals_123);  mul_93 = None
    add_100: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_94, primals_124);  mul_94 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_94: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_152: "f32[1568, 384]" = torch.ops.aten.view.default(add_100, [1568, 384]);  add_100 = None
    mm_13: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_152, permute_94)
    view_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_13, [8, 14, 14, 1152]);  mm_13 = None
    view_154: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_153, [8, 196, 3, 12, 32]);  view_153 = None
    permute_95: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_59: "f32[8, 12, 196, 32]" = unbind_5[0]
    getitem_60: "f32[8, 12, 196, 32]" = unbind_5[1]
    getitem_61: "f32[8, 12, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_96: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 3, 2]);  getitem_60 = None
    expand_28: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_59, [8, 12, 196, 32]);  getitem_59 = None
    clone_96: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_155: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_96, [96, 196, 32]);  clone_96 = None
    expand_29: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_96, [8, 12, 32, 196]);  permute_96 = None
    clone_97: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_156: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_97, [96, 32, 196]);  clone_97 = None
    bmm_14: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_155, view_156)
    view_157: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 12, 196, 196]);  bmm_14 = None
    mul_95: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_157, 0.1767766952966369);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_95, [-1], True)
    sub_31: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_95, amax_9);  mul_95 = amax_9 = None
    exp_9: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_10: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_12: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_98: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_30: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_98, [8, 12, 196, 196]);  clone_98 = None
    view_158: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_30, [96, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_61, [8, 12, 196, 32]);  getitem_61 = None
    clone_99: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_159: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_99, [96, 196, 32]);  clone_99 = None
    bmm_15: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_158, view_159)
    view_160: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_15, [8, 12, 196, 32]);  bmm_15 = None
    permute_97: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_100: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_161: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_100, [8, 14, 14, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_162: "f32[1568, 384]" = torch.ops.aten.view.default(view_161, [1568, 384]);  view_161 = None
    permute_98: "f32[384, 384]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_27: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_127, view_162, permute_98);  primals_127 = None
    view_163: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_27, [8, 14, 14, 384]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_101: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_101: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_98, clone_101);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_102: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_102, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 14, 14, 1]" = var_mean_22[0]
    getitem_63: "f32[8, 14, 14, 1]" = var_mean_22[1];  var_mean_22 = None
    add_102: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_22: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_32: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_63);  clone_102 = None
    mul_96: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
    mul_97: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_96, primals_128);  mul_96 = None
    add_103: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_97, primals_129);  mul_97 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_164: "f32[1568, 384]" = torch.ops.aten.view.default(add_103, [1568, 384]);  add_103 = None
    permute_99: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_28: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_131, view_164, permute_99);  primals_131 = None
    view_165: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_28, [8, 14, 14, 1152]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
    mul_99: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.7071067811865476)
    erf_9: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_104: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_100: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_98, add_104);  mul_98 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_103: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_166: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_103, [1568, 1152]);  clone_103 = None
    permute_100: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_29: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_133, view_166, permute_100);  primals_133 = None
    view_167: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_29, [8, 14, 14, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_104: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_167);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_105: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_101, clone_104);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_105: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_105, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_105, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 14, 14, 1]" = var_mean_23[0]
    getitem_65: "f32[8, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
    add_106: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_23: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_33: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_65);  clone_105 = None
    mul_101: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
    mul_102: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_101, primals_134);  mul_101 = None
    add_107: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_102, primals_135);  mul_102 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_101: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    view_168: "f32[1568, 384]" = torch.ops.aten.view.default(add_107, [1568, 384]);  add_107 = None
    mm_14: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_168, permute_101)
    view_169: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_14, [8, 14, 14, 1152]);  mm_14 = None
    view_170: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_169, [8, 196, 3, 12, 32]);  view_169 = None
    permute_102: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_170, [2, 0, 3, 1, 4]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_102);  permute_102 = None
    getitem_66: "f32[8, 12, 196, 32]" = unbind_6[0]
    getitem_67: "f32[8, 12, 196, 32]" = unbind_6[1]
    getitem_68: "f32[8, 12, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_103: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 3, 2]);  getitem_67 = None
    expand_32: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_66, [8, 12, 196, 32]);  getitem_66 = None
    clone_106: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_171: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_106, [96, 196, 32]);  clone_106 = None
    expand_33: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_103, [8, 12, 32, 196]);  permute_103 = None
    clone_107: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_172: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_107, [96, 32, 196]);  clone_107 = None
    bmm_16: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 12, 196, 196]);  bmm_16 = None
    mul_103: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_173, 0.1767766952966369);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_103, [-1], True)
    sub_34: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_103, amax_10);  mul_103 = amax_10 = None
    exp_10: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_11: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_13: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_108: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_34: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_108, [8, 12, 196, 196]);  clone_108 = None
    view_174: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_34, [96, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_68, [8, 12, 196, 32]);  getitem_68 = None
    clone_109: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_175: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_109, [96, 196, 32]);  clone_109 = None
    bmm_17: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_174, view_175)
    view_176: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_17, [8, 12, 196, 32]);  bmm_17 = None
    permute_104: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    clone_110: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_177: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_110, [8, 14, 14, 384]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_178: "f32[1568, 384]" = torch.ops.aten.view.default(view_177, [1568, 384]);  view_177 = None
    permute_105: "f32[384, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_30: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_138, view_178, permute_105);  primals_138 = None
    view_179: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_30, [8, 14, 14, 384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_111: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_108: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_105, clone_111);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_112: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_112, [3], correction = 0, keepdim = True)
    getitem_69: "f32[8, 14, 14, 1]" = var_mean_24[0]
    getitem_70: "f32[8, 14, 14, 1]" = var_mean_24[1];  var_mean_24 = None
    add_109: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
    rsqrt_24: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_35: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_70);  clone_112 = None
    mul_104: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
    mul_105: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_104, primals_139);  mul_104 = None
    add_110: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_105, primals_140);  mul_105 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[1568, 384]" = torch.ops.aten.view.default(add_110, [1568, 384]);  add_110 = None
    permute_106: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_31: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_142, view_180, permute_106);  primals_142 = None
    view_181: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_31, [8, 14, 14, 1152]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_107: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_10: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_111: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_108: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_106, add_111);  mul_106 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_113: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_113, [1568, 1152]);  clone_113 = None
    permute_107: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_32: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_144, view_182, permute_107);  primals_144 = None
    view_183: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_32, [8, 14, 14, 384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_114: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_112: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_108, clone_114);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_115: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_115, [3], correction = 0, keepdim = True)
    getitem_71: "f32[8, 14, 14, 1]" = var_mean_25[0]
    getitem_72: "f32[8, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
    add_113: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05);  getitem_71 = None
    rsqrt_25: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_36: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_115, getitem_72);  clone_115 = None
    mul_109: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
    mul_110: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_109, primals_145);  mul_109 = None
    add_114: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_110, primals_146);  mul_110 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_108: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_184: "f32[1568, 384]" = torch.ops.aten.view.default(add_114, [1568, 384]);  add_114 = None
    mm_15: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_184, permute_108)
    view_185: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_15, [8, 14, 14, 1152]);  mm_15 = None
    view_186: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_185, [8, 196, 3, 12, 32]);  view_185 = None
    permute_109: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_109);  permute_109 = None
    getitem_73: "f32[8, 12, 196, 32]" = unbind_7[0]
    getitem_74: "f32[8, 12, 196, 32]" = unbind_7[1]
    getitem_75: "f32[8, 12, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_110: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 3, 2]);  getitem_74 = None
    expand_36: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_73, [8, 12, 196, 32]);  getitem_73 = None
    clone_116: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_187: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_116, [96, 196, 32]);  clone_116 = None
    expand_37: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_110, [8, 12, 32, 196]);  permute_110 = None
    clone_117: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_188: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_117, [96, 32, 196]);  clone_117 = None
    bmm_18: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 12, 196, 196]);  bmm_18 = None
    mul_111: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_189, 0.1767766952966369);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_111, [-1], True)
    sub_37: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_111, amax_11);  mul_111 = amax_11 = None
    exp_11: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_12: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_14: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_118: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_38: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_118, [8, 12, 196, 196]);  clone_118 = None
    view_190: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_38, [96, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_75, [8, 12, 196, 32]);  getitem_75 = None
    clone_119: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_191: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_119, [96, 196, 32]);  clone_119 = None
    bmm_19: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_19, [8, 12, 196, 32]);  bmm_19 = None
    permute_111: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_120: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_193: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_120, [8, 14, 14, 384]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_194: "f32[1568, 384]" = torch.ops.aten.view.default(view_193, [1568, 384]);  view_193 = None
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_33: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_149, view_194, permute_112);  primals_149 = None
    view_195: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_33, [8, 14, 14, 384]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_121: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_115: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_112, clone_121);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_122: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_122, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 14, 14, 1]" = var_mean_26[0]
    getitem_77: "f32[8, 14, 14, 1]" = var_mean_26[1];  var_mean_26 = None
    add_116: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_26: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_38: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_122, getitem_77);  clone_122 = None
    mul_112: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
    mul_113: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_112, primals_150);  mul_112 = None
    add_117: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_113, primals_151);  mul_113 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_196: "f32[1568, 384]" = torch.ops.aten.view.default(add_117, [1568, 384]);  add_117 = None
    permute_113: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_34: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_153, view_196, permute_113);  primals_153 = None
    view_197: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_34, [8, 14, 14, 1152]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_114: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_115: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_11: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_118: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_116: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_114, add_118);  mul_114 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_123, [1568, 1152]);  clone_123 = None
    permute_114: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_35: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_155, view_198, permute_114);  primals_155 = None
    view_199: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_35, [8, 14, 14, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_124: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_119: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_115, clone_124);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_125: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_119, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_125, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 14, 14, 1]" = var_mean_27[0]
    getitem_79: "f32[8, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
    add_120: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_27: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_39: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_125, getitem_79);  clone_125 = None
    mul_117: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
    mul_118: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_117, primals_156);  mul_117 = None
    add_121: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_118, primals_157);  mul_118 = primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_115: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    view_200: "f32[1568, 384]" = torch.ops.aten.view.default(add_121, [1568, 384]);  add_121 = None
    mm_16: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_200, permute_115)
    view_201: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_16, [8, 14, 14, 1152]);  mm_16 = None
    view_202: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_201, [8, 196, 3, 12, 32]);  view_201 = None
    permute_116: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_202, [2, 0, 3, 1, 4]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
    getitem_80: "f32[8, 12, 196, 32]" = unbind_8[0]
    getitem_81: "f32[8, 12, 196, 32]" = unbind_8[1]
    getitem_82: "f32[8, 12, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_117: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_81, [0, 1, 3, 2]);  getitem_81 = None
    expand_40: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_80, [8, 12, 196, 32]);  getitem_80 = None
    clone_126: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_203: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_126, [96, 196, 32]);  clone_126 = None
    expand_41: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_117, [8, 12, 32, 196]);  permute_117 = None
    clone_127: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_204: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_127, [96, 32, 196]);  clone_127 = None
    bmm_20: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_203, view_204)
    view_205: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_20, [8, 12, 196, 196]);  bmm_20 = None
    mul_119: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_205, 0.1767766952966369);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_119, [-1], True)
    sub_40: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_119, amax_12);  mul_119 = amax_12 = None
    exp_12: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_13: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_15: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_128: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_42: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_128, [8, 12, 196, 196]);  clone_128 = None
    view_206: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_42, [96, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_82, [8, 12, 196, 32]);  getitem_82 = None
    clone_129: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_207: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_129, [96, 196, 32]);  clone_129 = None
    bmm_21: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_206, view_207)
    view_208: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_21, [8, 12, 196, 32]);  bmm_21 = None
    permute_118: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_130: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_209: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_130, [8, 14, 14, 384]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_210: "f32[1568, 384]" = torch.ops.aten.view.default(view_209, [1568, 384]);  view_209 = None
    permute_119: "f32[384, 384]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_36: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_160, view_210, permute_119);  primals_160 = None
    view_211: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_36, [8, 14, 14, 384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_131: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_211);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_122: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_119, clone_131);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_132: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_132, [3], correction = 0, keepdim = True)
    getitem_83: "f32[8, 14, 14, 1]" = var_mean_28[0]
    getitem_84: "f32[8, 14, 14, 1]" = var_mean_28[1];  var_mean_28 = None
    add_123: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
    rsqrt_28: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_41: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_132, getitem_84);  clone_132 = None
    mul_120: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
    mul_121: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_120, primals_161);  mul_120 = None
    add_124: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_121, primals_162);  mul_121 = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_212: "f32[1568, 384]" = torch.ops.aten.view.default(add_124, [1568, 384]);  add_124 = None
    permute_120: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_37: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_164, view_212, permute_120);  primals_164 = None
    view_213: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_37, [8, 14, 14, 1152]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_12: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_125: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_124: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_122, add_125);  mul_122 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_214: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_133, [1568, 1152]);  clone_133 = None
    permute_121: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_38: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_166, view_214, permute_121);  primals_166 = None
    view_215: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_38, [8, 14, 14, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_134: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_126: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_122, clone_134);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_135: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_135, [3], correction = 0, keepdim = True)
    getitem_85: "f32[8, 14, 14, 1]" = var_mean_29[0]
    getitem_86: "f32[8, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
    add_127: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_85, 1e-05);  getitem_85 = None
    rsqrt_29: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_42: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_135, getitem_86);  clone_135 = None
    mul_125: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
    mul_126: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_125, primals_167);  mul_125 = None
    add_128: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_126, primals_168);  mul_126 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_122: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    view_216: "f32[1568, 384]" = torch.ops.aten.view.default(add_128, [1568, 384]);  add_128 = None
    mm_17: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_216, permute_122)
    view_217: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_17, [8, 14, 14, 1152]);  mm_17 = None
    view_218: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_217, [8, 196, 3, 12, 32]);  view_217 = None
    permute_123: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_218, [2, 0, 3, 1, 4]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_123);  permute_123 = None
    getitem_87: "f32[8, 12, 196, 32]" = unbind_9[0]
    getitem_88: "f32[8, 12, 196, 32]" = unbind_9[1]
    getitem_89: "f32[8, 12, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_124: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_88, [0, 1, 3, 2]);  getitem_88 = None
    expand_44: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_87, [8, 12, 196, 32]);  getitem_87 = None
    clone_136: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_219: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_136, [96, 196, 32]);  clone_136 = None
    expand_45: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_124, [8, 12, 32, 196]);  permute_124 = None
    clone_137: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_220: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_137, [96, 32, 196]);  clone_137 = None
    bmm_22: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_219, view_220)
    view_221: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_22, [8, 12, 196, 196]);  bmm_22 = None
    mul_127: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_221, 0.1767766952966369);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_127, [-1], True)
    sub_43: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_127, amax_13);  mul_127 = amax_13 = None
    exp_13: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_14: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_16: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_138: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_46: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_138, [8, 12, 196, 196]);  clone_138 = None
    view_222: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_46, [96, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_89, [8, 12, 196, 32]);  getitem_89 = None
    clone_139: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_223: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_139, [96, 196, 32]);  clone_139 = None
    bmm_23: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_222, view_223)
    view_224: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_23, [8, 12, 196, 32]);  bmm_23 = None
    permute_125: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_140: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_225: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_140, [8, 14, 14, 384]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_226: "f32[1568, 384]" = torch.ops.aten.view.default(view_225, [1568, 384]);  view_225 = None
    permute_126: "f32[384, 384]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_39: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_171, view_226, permute_126);  primals_171 = None
    view_227: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_39, [8, 14, 14, 384]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_141: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_227);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_129: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_126, clone_141);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_142: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_142, [3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 14, 14, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 14, 14, 1]" = var_mean_30[1];  var_mean_30 = None
    add_130: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_30: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_44: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_142, getitem_91);  clone_142 = None
    mul_128: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
    mul_129: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_128, primals_172);  mul_128 = None
    add_131: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_129, primals_173);  mul_129 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_228: "f32[1568, 384]" = torch.ops.aten.view.default(add_131, [1568, 384]);  add_131 = None
    permute_127: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_40: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_175, view_228, permute_127);  primals_175 = None
    view_229: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_40, [8, 14, 14, 1152]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
    mul_131: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476)
    erf_13: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_132: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_132: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_130, add_132);  mul_130 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_143: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_230: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_143, [1568, 1152]);  clone_143 = None
    permute_128: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_41: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_177, view_230, permute_128);  primals_177 = None
    view_231: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_41, [8, 14, 14, 384]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_144: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_231);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_133: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_129, clone_144);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_145: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_145, [3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 14, 14, 1]" = var_mean_31[0]
    getitem_93: "f32[8, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
    add_134: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_31: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_45: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_145, getitem_93);  clone_145 = None
    mul_133: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
    mul_134: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_133, primals_178);  mul_133 = None
    add_135: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_134, primals_179);  mul_134 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_129: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_232: "f32[1568, 384]" = torch.ops.aten.view.default(add_135, [1568, 384]);  add_135 = None
    mm_18: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_232, permute_129)
    view_233: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_18, [8, 14, 14, 1152]);  mm_18 = None
    view_234: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_233, [8, 196, 3, 12, 32]);  view_233 = None
    permute_130: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
    getitem_94: "f32[8, 12, 196, 32]" = unbind_10[0]
    getitem_95: "f32[8, 12, 196, 32]" = unbind_10[1]
    getitem_96: "f32[8, 12, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_131: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_95, [0, 1, 3, 2]);  getitem_95 = None
    expand_48: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_94, [8, 12, 196, 32]);  getitem_94 = None
    clone_146: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_235: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_146, [96, 196, 32]);  clone_146 = None
    expand_49: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_131, [8, 12, 32, 196]);  permute_131 = None
    clone_147: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_236: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_147, [96, 32, 196]);  clone_147 = None
    bmm_24: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_235, view_236)
    view_237: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_24, [8, 12, 196, 196]);  bmm_24 = None
    mul_135: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_237, 0.1767766952966369);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_135, [-1], True)
    sub_46: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_135, amax_14);  mul_135 = amax_14 = None
    exp_14: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_15: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_17: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_148: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_50: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_148, [8, 12, 196, 196]);  clone_148 = None
    view_238: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_50, [96, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_96, [8, 12, 196, 32]);  getitem_96 = None
    clone_149: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_239: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_149, [96, 196, 32]);  clone_149 = None
    bmm_25: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_238, view_239)
    view_240: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_25, [8, 12, 196, 32]);  bmm_25 = None
    permute_132: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    clone_150: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_241: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_150, [8, 14, 14, 384]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_242: "f32[1568, 384]" = torch.ops.aten.view.default(view_241, [1568, 384]);  view_241 = None
    permute_133: "f32[384, 384]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_42: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_182, view_242, permute_133);  primals_182 = None
    view_243: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_42, [8, 14, 14, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_151: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_243);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_136: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_133, clone_151);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_152: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_152, [3], correction = 0, keepdim = True)
    getitem_97: "f32[8, 14, 14, 1]" = var_mean_32[0]
    getitem_98: "f32[8, 14, 14, 1]" = var_mean_32[1];  var_mean_32 = None
    add_137: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
    rsqrt_32: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_47: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_152, getitem_98);  clone_152 = None
    mul_136: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
    mul_137: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_136, primals_183);  mul_136 = None
    add_138: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_137, primals_184);  mul_137 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_244: "f32[1568, 384]" = torch.ops.aten.view.default(add_138, [1568, 384]);  add_138 = None
    permute_134: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_43: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_186, view_244, permute_134);  primals_186 = None
    view_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_43, [8, 14, 14, 1152]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_138: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.5)
    mul_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476)
    erf_14: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_140: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_138, add_139);  mul_138 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_153, [1568, 1152]);  clone_153 = None
    permute_135: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_44: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_188, view_246, permute_135);  primals_188 = None
    view_247: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_44, [8, 14, 14, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_247);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_140: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_136, clone_154);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_155: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_155, [3], correction = 0, keepdim = True)
    getitem_99: "f32[8, 14, 14, 1]" = var_mean_33[0]
    getitem_100: "f32[8, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
    add_141: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
    rsqrt_33: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_48: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_155, getitem_100);  clone_155 = None
    mul_141: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
    mul_142: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_141, primals_189);  mul_141 = None
    add_142: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_142, primals_190);  mul_142 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_136: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    view_248: "f32[1568, 384]" = torch.ops.aten.view.default(add_142, [1568, 384]);  add_142 = None
    mm_19: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_248, permute_136)
    view_249: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_19, [8, 14, 14, 1152]);  mm_19 = None
    view_250: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_249, [8, 196, 3, 12, 32]);  view_249 = None
    permute_137: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_250, [2, 0, 3, 1, 4]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
    getitem_101: "f32[8, 12, 196, 32]" = unbind_11[0]
    getitem_102: "f32[8, 12, 196, 32]" = unbind_11[1]
    getitem_103: "f32[8, 12, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_138: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_102, [0, 1, 3, 2]);  getitem_102 = None
    expand_52: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_101, [8, 12, 196, 32]);  getitem_101 = None
    clone_156: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_251: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_156, [96, 196, 32]);  clone_156 = None
    expand_53: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_138, [8, 12, 32, 196]);  permute_138 = None
    clone_157: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_252: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_157, [96, 32, 196]);  clone_157 = None
    bmm_26: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_26, [8, 12, 196, 196]);  bmm_26 = None
    mul_143: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_253, 0.1767766952966369);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_143, [-1], True)
    sub_49: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_143, amax_15);  mul_143 = amax_15 = None
    exp_15: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_16: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_18: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_158: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_54: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_158, [8, 12, 196, 196]);  clone_158 = None
    view_254: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_54, [96, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_103, [8, 12, 196, 32]);  getitem_103 = None
    clone_159: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_255: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_159, [96, 196, 32]);  clone_159 = None
    bmm_27: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_27, [8, 12, 196, 32]);  bmm_27 = None
    permute_139: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_160: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_257: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_160, [8, 14, 14, 384]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_258: "f32[1568, 384]" = torch.ops.aten.view.default(view_257, [1568, 384]);  view_257 = None
    permute_140: "f32[384, 384]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_45: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_193, view_258, permute_140);  primals_193 = None
    view_259: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_45, [8, 14, 14, 384]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_161: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_259);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_143: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_140, clone_161);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_162: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_162, [3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 14, 14, 1]" = var_mean_34[0]
    getitem_105: "f32[8, 14, 14, 1]" = var_mean_34[1];  var_mean_34 = None
    add_144: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_34: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_50: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_162, getitem_105);  clone_162 = None
    mul_144: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
    mul_145: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_144, primals_194);  mul_144 = None
    add_145: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_145, primals_195);  mul_145 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_260: "f32[1568, 384]" = torch.ops.aten.view.default(add_145, [1568, 384]);  add_145 = None
    permute_141: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_46: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_197, view_260, permute_141);  primals_197 = None
    view_261: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_46, [8, 14, 14, 1152]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_146: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_147: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_15: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_146: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_148: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_146, add_146);  mul_146 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_163: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_262: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_163, [1568, 1152]);  clone_163 = None
    permute_142: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_47: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_199, view_262, permute_142);  primals_199 = None
    view_263: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_47, [8, 14, 14, 384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_164: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_263);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_147: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_143, clone_164);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_165: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_147, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_165, [3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 14, 14, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
    add_148: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_35: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_51: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_107);  clone_165 = None
    mul_149: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
    mul_150: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_149, primals_200);  mul_149 = None
    add_149: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_150, primals_201);  mul_150 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_143: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    view_264: "f32[1568, 384]" = torch.ops.aten.view.default(add_149, [1568, 384]);  add_149 = None
    mm_20: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_264, permute_143)
    view_265: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_20, [8, 14, 14, 1152]);  mm_20 = None
    view_266: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_265, [8, 196, 3, 12, 32]);  view_265 = None
    permute_144: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_266, [2, 0, 3, 1, 4]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_144);  permute_144 = None
    getitem_108: "f32[8, 12, 196, 32]" = unbind_12[0]
    getitem_109: "f32[8, 12, 196, 32]" = unbind_12[1]
    getitem_110: "f32[8, 12, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_145: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_109, [0, 1, 3, 2]);  getitem_109 = None
    expand_56: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_108, [8, 12, 196, 32]);  getitem_108 = None
    clone_166: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_267: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_166, [96, 196, 32]);  clone_166 = None
    expand_57: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_145, [8, 12, 32, 196]);  permute_145 = None
    clone_167: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_268: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_167, [96, 32, 196]);  clone_167 = None
    bmm_28: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_267, view_268)
    view_269: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_28, [8, 12, 196, 196]);  bmm_28 = None
    mul_151: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_269, 0.1767766952966369);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_151, [-1], True)
    sub_52: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_151, amax_16);  mul_151 = amax_16 = None
    exp_16: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_17: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_19: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_168: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_58: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_168, [8, 12, 196, 196]);  clone_168 = None
    view_270: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_58, [96, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_110, [8, 12, 196, 32]);  getitem_110 = None
    clone_169: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_271: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_169, [96, 196, 32]);  clone_169 = None
    bmm_29: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_270, view_271)
    view_272: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_29, [8, 12, 196, 32]);  bmm_29 = None
    permute_146: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_170: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_273: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_170, [8, 14, 14, 384]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_274: "f32[1568, 384]" = torch.ops.aten.view.default(view_273, [1568, 384]);  view_273 = None
    permute_147: "f32[384, 384]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_48: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_204, view_274, permute_147);  primals_204 = None
    view_275: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_48, [8, 14, 14, 384]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_171: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_275);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_150: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_147, clone_171);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_172: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_172, [3], correction = 0, keepdim = True)
    getitem_111: "f32[8, 14, 14, 1]" = var_mean_36[0]
    getitem_112: "f32[8, 14, 14, 1]" = var_mean_36[1];  var_mean_36 = None
    add_151: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
    rsqrt_36: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_53: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_172, getitem_112);  clone_172 = None
    mul_152: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
    mul_153: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_152, primals_205);  mul_152 = None
    add_152: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_153, primals_206);  mul_153 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1568, 384]" = torch.ops.aten.view.default(add_152, [1568, 384]);  add_152 = None
    permute_148: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_49: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_208, view_276, permute_148);  primals_208 = None
    view_277: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_49, [8, 14, 14, 1152]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
    mul_155: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476)
    erf_16: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_156: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_154, add_153);  mul_154 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_173: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_278: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_173, [1568, 1152]);  clone_173 = None
    permute_149: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_50: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_210, view_278, permute_149);  primals_210 = None
    view_279: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_50, [8, 14, 14, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_174: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_279);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_154: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_150, clone_174);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_175: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_154, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_175, [3], correction = 0, keepdim = True)
    getitem_113: "f32[8, 14, 14, 1]" = var_mean_37[0]
    getitem_114: "f32[8, 14, 14, 1]" = var_mean_37[1];  var_mean_37 = None
    add_155: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-05);  getitem_113 = None
    rsqrt_37: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_54: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_175, getitem_114);  clone_175 = None
    mul_157: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
    mul_158: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_157, primals_211);  mul_157 = None
    add_156: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_158, primals_212);  mul_158 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_150: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    view_280: "f32[1568, 384]" = torch.ops.aten.view.default(add_156, [1568, 384]);  add_156 = None
    mm_21: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_280, permute_150)
    view_281: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_21, [8, 14, 14, 1152]);  mm_21 = None
    view_282: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_281, [8, 196, 3, 12, 32]);  view_281 = None
    permute_151: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_282, [2, 0, 3, 1, 4]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_151);  permute_151 = None
    getitem_115: "f32[8, 12, 196, 32]" = unbind_13[0]
    getitem_116: "f32[8, 12, 196, 32]" = unbind_13[1]
    getitem_117: "f32[8, 12, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_152: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 3, 2]);  getitem_116 = None
    expand_60: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_115, [8, 12, 196, 32]);  getitem_115 = None
    clone_176: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_283: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_176, [96, 196, 32]);  clone_176 = None
    expand_61: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_152, [8, 12, 32, 196]);  permute_152 = None
    clone_177: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_284: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_177, [96, 32, 196]);  clone_177 = None
    bmm_30: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_283, view_284)
    view_285: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_30, [8, 12, 196, 196]);  bmm_30 = None
    mul_159: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_285, 0.1767766952966369);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_159, [-1], True)
    sub_55: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_159, amax_17);  mul_159 = amax_17 = None
    exp_17: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_18: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_20: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_178: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_62: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_178, [8, 12, 196, 196]);  clone_178 = None
    view_286: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_62, [96, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 12, 196, 32]);  getitem_117 = None
    clone_179: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_287: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_179, [96, 196, 32]);  clone_179 = None
    bmm_31: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_286, view_287)
    view_288: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_31, [8, 12, 196, 32]);  bmm_31 = None
    permute_153: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    clone_180: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_289: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_180, [8, 14, 14, 384]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_290: "f32[1568, 384]" = torch.ops.aten.view.default(view_289, [1568, 384]);  view_289 = None
    permute_154: "f32[384, 384]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_51: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_215, view_290, permute_154);  primals_215 = None
    view_291: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_51, [8, 14, 14, 384]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_181: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_291);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_157: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_154, clone_181);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_182: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_182, [3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 14, 14, 1]" = var_mean_38[0]
    getitem_119: "f32[8, 14, 14, 1]" = var_mean_38[1];  var_mean_38 = None
    add_158: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_38: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_56: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_182, getitem_119);  clone_182 = None
    mul_160: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
    mul_161: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_160, primals_216);  mul_160 = None
    add_159: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_161, primals_217);  mul_161 = primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_292: "f32[1568, 384]" = torch.ops.aten.view.default(add_159, [1568, 384]);  add_159 = None
    permute_155: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_52: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_219, view_292, permute_155);  primals_219 = None
    view_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_52, [8, 14, 14, 1152]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_162: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.5)
    mul_163: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.7071067811865476)
    erf_17: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_160: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_164: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_162, add_160);  mul_162 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_183: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_294: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_183, [1568, 1152]);  clone_183 = None
    permute_156: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_53: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_221, view_294, permute_156);  primals_221 = None
    view_295: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_53, [8, 14, 14, 384]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_184: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_295);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_161: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_157, clone_184);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    view_296: "f32[8, 196, 384]" = torch.ops.aten.view.default(add_161, [8, 196, 384]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    expand_64: "f32[8, 1, 384]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    cat: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand_64, view_296], 1);  expand_64 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_9: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807)
    slice_10: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_39 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 197, 1]" = var_mean_39[0]
    getitem_121: "f32[8, 197, 1]" = var_mean_39[1];  var_mean_39 = None
    add_162: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_39: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat, getitem_121)
    mul_165: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
    mul_166: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_165, primals_222);  mul_165 = None
    add_163: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_166, primals_223);  mul_166 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_157: "f32[384, 768]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    view_297: "f32[1576, 384]" = torch.ops.aten.view.default(add_163, [1576, 384])
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_297, permute_157)
    view_298: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    view_299: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_298, [8, 197, 2, 12, 32]);  view_298 = None
    permute_158: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_299, [2, 0, 3, 1, 4]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_158);  permute_158 = None
    getitem_122: "f32[8, 12, 197, 32]" = unbind_14[0]
    getitem_123: "f32[8, 12, 197, 32]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_11: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_163, 0, 0, 9223372036854775807);  add_163 = None
    slice_12: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 1);  slice_11 = None
    slice_13: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_12, 2, 0, 9223372036854775807);  slice_12 = None
    permute_159: "f32[384, 384]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    view_300: "f32[8, 384]" = torch.ops.aten.view.default(slice_13, [8, 384]);  slice_13 = None
    mm_23: "f32[8, 384]" = torch.ops.aten.mm.default(view_300, permute_159)
    view_301: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_23, [8, 1, 384]);  mm_23 = None
    view_302: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_301, [8, 12, 1, 32]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_167: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_302, 0.1767766952966369);  view_302 = None
    permute_160: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_122, [0, 1, 3, 2]);  getitem_122 = None
    expand_65: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_167, [8, 12, 1, 32]);  mul_167 = None
    view_303: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_65, [96, 1, 32]);  expand_65 = None
    expand_66: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_160, [8, 12, 32, 197]);  permute_160 = None
    clone_185: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_304: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_185, [96, 32, 197]);  clone_185 = None
    bmm_32: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_303, view_304)
    view_305: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_32, [8, 12, 1, 197]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_305, [-1], True)
    sub_58: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_305, amax_18);  view_305 = amax_18 = None
    exp_18: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_19: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_21: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    clone_186: "f32[8, 12, 1, 197]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_67: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(clone_186, [8, 12, 1, 197]);  clone_186 = None
    view_306: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_67, [96, 1, 197]);  expand_67 = None
    expand_68: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_123, [8, 12, 197, 32]);  getitem_123 = None
    clone_187: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_307: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_187, [96, 197, 32]);  clone_187 = None
    bmm_33: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_306, view_307)
    view_308: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_33, [8, 12, 1, 32]);  bmm_33 = None
    permute_161: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    view_309: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_161, [8, 1, 384]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_310: "f32[8, 384]" = torch.ops.aten.view.default(view_309, [8, 384]);  view_309 = None
    permute_162: "f32[384, 384]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_54: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_227, view_310, permute_162);  primals_227 = None
    view_311: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_54, [8, 1, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    clone_188: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_311);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_164: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_10, clone_188);  slice_10 = clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_40 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 1, 1]" = var_mean_40[0]
    getitem_125: "f32[8, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_165: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_40: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_59: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_164, getitem_125)
    mul_168: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
    mul_169: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_168, primals_228);  mul_168 = None
    add_166: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_169, primals_229);  mul_169 = primals_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_312: "f32[8, 384]" = torch.ops.aten.view.default(add_166, [8, 384]);  add_166 = None
    permute_163: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_55: "f32[8, 1152]" = torch.ops.aten.addmm.default(primals_231, view_312, permute_163);  primals_231 = None
    view_313: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_55, [8, 1, 1152]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_170: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.5)
    mul_171: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476)
    erf_18: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_167: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_172: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_170, add_167);  mul_170 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_189: "f32[8, 1, 1152]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_314: "f32[8, 1152]" = torch.ops.aten.view.default(clone_189, [8, 1152]);  clone_189 = None
    permute_164: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_56: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_233, view_314, permute_164);  primals_233 = None
    view_315: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_56, [8, 1, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_190: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_315);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_168: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_164, clone_190);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_14: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807)
    slice_15: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_14, 1, 1, 9223372036854775807);  slice_14 = None
    cat_1: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_168, slice_15], 1);  add_168 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_16: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
    slice_17: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 1);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_41 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 197, 1]" = var_mean_41[0]
    getitem_127: "f32[8, 197, 1]" = var_mean_41[1];  var_mean_41 = None
    add_169: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_41: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_60: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_127)
    mul_173: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
    mul_174: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_173, primals_234);  mul_173 = None
    add_170: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_174, primals_235);  mul_174 = primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_165: "f32[384, 768]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    view_316: "f32[1576, 384]" = torch.ops.aten.view.default(add_170, [1576, 384])
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_316, permute_165)
    view_317: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    view_318: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_317, [8, 197, 2, 12, 32]);  view_317 = None
    permute_166: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_318, [2, 0, 3, 1, 4]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_166);  permute_166 = None
    getitem_128: "f32[8, 12, 197, 32]" = unbind_15[0]
    getitem_129: "f32[8, 12, 197, 32]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_18: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_170, 0, 0, 9223372036854775807);  add_170 = None
    slice_19: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_18, 1, 0, 1);  slice_18 = None
    slice_20: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, 9223372036854775807);  slice_19 = None
    permute_167: "f32[384, 384]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    view_319: "f32[8, 384]" = torch.ops.aten.view.default(slice_20, [8, 384]);  slice_20 = None
    mm_25: "f32[8, 384]" = torch.ops.aten.mm.default(view_319, permute_167)
    view_320: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_25, [8, 1, 384]);  mm_25 = None
    view_321: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_320, [8, 12, 1, 32]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_175: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_321, 0.1767766952966369);  view_321 = None
    permute_168: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
    expand_69: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_175, [8, 12, 1, 32]);  mul_175 = None
    view_322: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_69, [96, 1, 32]);  expand_69 = None
    expand_70: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_168, [8, 12, 32, 197]);  permute_168 = None
    clone_191: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_323: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_191, [96, 32, 197]);  clone_191 = None
    bmm_34: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_322, view_323)
    view_324: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_34, [8, 12, 1, 197]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_324, [-1], True)
    sub_61: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_324, amax_19);  view_324 = amax_19 = None
    exp_19: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_20: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_22: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    clone_192: "f32[8, 12, 1, 197]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_71: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(clone_192, [8, 12, 1, 197]);  clone_192 = None
    view_325: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_71, [96, 1, 197]);  expand_71 = None
    expand_72: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_129, [8, 12, 197, 32]);  getitem_129 = None
    clone_193: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_326: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_193, [96, 197, 32]);  clone_193 = None
    bmm_35: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_325, view_326)
    view_327: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_35, [8, 12, 1, 32]);  bmm_35 = None
    permute_169: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    view_328: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_169, [8, 1, 384]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_329: "f32[8, 384]" = torch.ops.aten.view.default(view_328, [8, 384]);  view_328 = None
    permute_170: "f32[384, 384]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_57: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_239, view_329, permute_170);  primals_239 = None
    view_330: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_57, [8, 1, 384]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    clone_194: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_330);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_171: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_17, clone_194);  slice_17 = clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_42 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 1, 1]" = var_mean_42[0]
    getitem_131: "f32[8, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_172: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_42: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_62: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_171, getitem_131)
    mul_176: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
    mul_177: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_176, primals_240);  mul_176 = None
    add_173: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_177, primals_241);  mul_177 = primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_331: "f32[8, 384]" = torch.ops.aten.view.default(add_173, [8, 384]);  add_173 = None
    permute_171: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_58: "f32[8, 1152]" = torch.ops.aten.addmm.default(primals_243, view_331, permute_171);  primals_243 = None
    view_332: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_58, [8, 1, 1152]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_178: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
    mul_179: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476)
    erf_19: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_174: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_180: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_178, add_174);  mul_178 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_195: "f32[8, 1, 1152]" = torch.ops.aten.clone.default(mul_180);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_333: "f32[8, 1152]" = torch.ops.aten.view.default(clone_195, [8, 1152]);  clone_195 = None
    permute_172: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_59: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_245, view_333, permute_172);  primals_245 = None
    view_334: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_59, [8, 1, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_196: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_334);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_175: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_171, clone_196);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_21: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
    slice_22: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_21, 1, 1, 9223372036854775807);  slice_21 = None
    cat_2: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_175, slice_22], 1);  add_175 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 197, 1]" = var_mean_43[0]
    getitem_133: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
    add_176: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_63: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_133)
    mul_181: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
    mul_182: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_181, primals_246);  mul_181 = None
    add_177: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_182, primals_247);  mul_182 = primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    slice_23: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_177, 0, 0, 9223372036854775807)
    select: "f32[8, 384]" = torch.ops.aten.select.int(slice_23, 1, 0);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:713, code: x = self.head_drop(x)
    clone_197: "f32[8, 197, 384]" = torch.ops.aten.clone.default(add_177);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    permute_173: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_60: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_249, select, permute_173);  primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    slice_24: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(clone_197, 0, 0, 9223372036854775807);  clone_197 = None
    slice_25: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_24, 1, 1, 9223372036854775807);  slice_24 = None
    permute_174: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_25, memory_format = torch.contiguous_format);  slice_25 = None
    view_335: "f32[1568, 384]" = torch.ops.aten.view.default(clone_198, [1568, 384]);  clone_198 = None
    mm_26: "f32[1568, 1000]" = torch.ops.aten.mm.default(view_335, permute_174)
    view_336: "f32[8, 196, 1000]" = torch.ops.aten.view.default(mm_26, [8, 196, 1000]);  mm_26 = None
    add_178: "f32[8, 196, 1000]" = torch.ops.aten.add.Tensor(view_336, primals_251);  view_336 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    max_1 = torch.ops.aten.max.dim(add_178, 1);  add_178 = None
    getitem_134: "f32[8, 1000]" = max_1[0]
    getitem_135: "i64[8, 1000]" = max_1[1];  max_1 = None
    mul_183: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(getitem_134, 0.5);  getitem_134 = None
    add_179: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm_60, mul_183);  addmm_60 = mul_183 = None
    mul_184: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(tangents_1, 0.5)
    unsqueeze_60: "f32[8, 1, 1000]" = torch.ops.aten.unsqueeze.default(mul_184, 1);  mul_184 = None
    unsqueeze_61: "i64[8, 1, 1000]" = torch.ops.aten.unsqueeze.default(getitem_135, 1);  getitem_135 = None
    full_4: "f32[8, 196, 1000]" = torch.ops.aten.full.default([8, 196, 1000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[8, 196, 1000]" = torch.ops.aten.scatter.src(full_4, 1, unsqueeze_61, unsqueeze_60);  full_4 = unsqueeze_61 = unsqueeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    sum_21: "f32[1, 1, 1000]" = torch.ops.aten.sum.dim_IntList(scatter, [0, 1], True)
    view_337: "f32[1000]" = torch.ops.aten.view.default(sum_21, [1000]);  sum_21 = None
    view_338: "f32[1568, 1000]" = torch.ops.aten.view.default(scatter, [1568, 1000]);  scatter = None
    permute_175: "f32[1000, 1568]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_27: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_175, view_335);  permute_175 = view_335 = None
    permute_176: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    permute_177: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    mm_28: "f32[1568, 384]" = torch.ops.aten.mm.default(view_338, permute_177);  view_338 = permute_177 = None
    view_339: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_28, [8, 196, 384]);  mm_28 = None
    permute_178: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    full_5: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_5, view_339, 1, 1, 9223372036854775807);  full_5 = view_339 = None
    full_6: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter, 0, 0, 9223372036854775807);  full_6 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    permute_179: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_29: "f32[8, 384]" = torch.ops.aten.mm.default(tangents_1, permute_179);  permute_179 = None
    permute_180: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_30: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_180, select);  permute_180 = select = None
    permute_181: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_22: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_340: "f32[1000]" = torch.ops.aten.view.default(sum_22, [1000]);  sum_22 = None
    permute_182: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    full_7: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 197, 384]" = torch.ops.aten.select_scatter.default(full_7, mm_29, 1, 0);  full_7 = mm_29 = None
    full_8: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_8, select_scatter, 0, 0, 9223372036854775807);  full_8 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    add_180: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_1, slice_scatter_2);  slice_scatter_1 = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    sub_64: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_133);  cat_2 = getitem_133 = None
    mul_185: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_43);  sub_64 = None
    mul_186: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_180, primals_246);  primals_246 = None
    mul_187: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_186, 384)
    sum_23: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
    mul_188: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_186, mul_185);  mul_186 = None
    sum_24: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
    mul_189: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_185, sum_24);  sum_24 = None
    sub_65: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_187, sum_23);  mul_187 = sum_23 = None
    sub_66: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_65, mul_189);  sub_65 = mul_189 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 384);  rsqrt_43 = None
    mul_190: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_20, sub_66);  div_20 = sub_66 = None
    mul_191: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_180, mul_185);  mul_185 = None
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_26: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(mul_190, 1, 0, 1)
    slice_27: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(mul_190, 1, 1, 197);  mul_190 = None
    full_9: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_3: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_9, slice_27, 1, 1, 9223372036854775807);  full_9 = slice_27 = None
    full_10: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_3, 0, 0, 9223372036854775807);  full_10 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_341: "f32[8, 384]" = torch.ops.aten.view.default(slice_26, [8, 384])
    permute_183: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_31: "f32[8, 1152]" = torch.ops.aten.mm.default(view_341, permute_183);  permute_183 = None
    permute_184: "f32[384, 8]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_32: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_184, view_333);  permute_184 = view_333 = None
    permute_185: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_27: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[384]" = torch.ops.aten.view.default(sum_27, [384]);  sum_27 = None
    permute_186: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_343: "f32[8, 1, 1152]" = torch.ops.aten.view.default(mm_31, [8, 1, 1152]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_192: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476)
    erf_20: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_181: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_193: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
    mul_194: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, view_332)
    mul_195: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
    exp_20: "f32[8, 1, 1152]" = torch.ops.aten.exp.default(mul_195);  mul_195 = None
    mul_196: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_197: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, mul_196);  view_332 = mul_196 = None
    add_182: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
    mul_198: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_343, add_182);  view_343 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_344: "f32[8, 1152]" = torch.ops.aten.view.default(mul_198, [8, 1152]);  mul_198 = None
    permute_187: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    mm_33: "f32[8, 384]" = torch.ops.aten.mm.default(view_344, permute_187);  permute_187 = None
    permute_188: "f32[1152, 8]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_34: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_188, view_331);  permute_188 = view_331 = None
    permute_189: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_28: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[1152]" = torch.ops.aten.view.default(sum_28, [1152]);  sum_28 = None
    permute_190: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_346: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_33, [8, 1, 384]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    sub_67: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_171, getitem_131);  add_171 = getitem_131 = None
    mul_199: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_42);  sub_67 = None
    mul_200: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_346, primals_240);  primals_240 = None
    mul_201: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_200, 384)
    sum_29: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
    mul_202: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_200, mul_199);  mul_200 = None
    sum_30: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
    mul_203: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_199, sum_30);  sum_30 = None
    sub_68: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(mul_201, sum_29);  mul_201 = sum_29 = None
    sub_69: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(sub_68, mul_203);  sub_68 = mul_203 = None
    div_21: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 384);  rsqrt_42 = None
    mul_204: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(div_21, sub_69);  div_21 = sub_69 = None
    mul_205: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_346, mul_199);  mul_199 = None
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_346, [0, 1]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_183: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_26, mul_204);  slice_26 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_347: "f32[8, 384]" = torch.ops.aten.view.default(add_183, [8, 384])
    permute_191: "f32[384, 384]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    mm_35: "f32[8, 384]" = torch.ops.aten.mm.default(view_347, permute_191);  permute_191 = None
    permute_192: "f32[384, 8]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_36: "f32[384, 384]" = torch.ops.aten.mm.default(permute_192, view_329);  permute_192 = view_329 = None
    permute_193: "f32[384, 384]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_33: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[384]" = torch.ops.aten.view.default(sum_33, [384]);  sum_33 = None
    permute_194: "f32[384, 384]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_349: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_35, [8, 1, 384]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    view_350: "f32[8, 1, 12, 32]" = torch.ops.aten.view.default(view_349, [8, 1, 12, 32]);  view_349 = None
    permute_195: "f32[8, 12, 1, 32]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    view_351: "f32[96, 1, 32]" = torch.ops.aten.view.default(permute_195, [96, 1, 32]);  permute_195 = None
    permute_196: "f32[96, 197, 1]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    bmm_36: "f32[96, 197, 32]" = torch.ops.aten.bmm.default(permute_196, view_351);  permute_196 = None
    permute_197: "f32[96, 32, 197]" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
    bmm_37: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_351, permute_197);  view_351 = permute_197 = None
    view_352: "f32[8, 12, 197, 32]" = torch.ops.aten.view.default(bmm_36, [8, 12, 197, 32]);  bmm_36 = None
    view_353: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_37, [8, 12, 1, 197]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    alias_23: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_206: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(view_353, alias_23);  view_353 = None
    sum_34: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [-1], True)
    mul_207: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(alias_23, sum_34);  alias_23 = sum_34 = None
    sub_70: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    view_354: "f32[96, 1, 197]" = torch.ops.aten.view.default(sub_70, [96, 1, 197]);  sub_70 = None
    permute_198: "f32[96, 32, 1]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    bmm_38: "f32[96, 32, 197]" = torch.ops.aten.bmm.default(permute_198, view_354);  permute_198 = None
    permute_199: "f32[96, 197, 32]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    bmm_39: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_354, permute_199);  view_354 = permute_199 = None
    view_355: "f32[8, 12, 32, 197]" = torch.ops.aten.view.default(bmm_38, [8, 12, 32, 197]);  bmm_38 = None
    view_356: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_39, [8, 12, 1, 32]);  bmm_39 = None
    permute_200: "f32[8, 12, 197, 32]" = torch.ops.aten.permute.default(view_355, [0, 1, 3, 2]);  view_355 = None
    mul_208: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_356, 0.1767766952966369);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    view_357: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_208, [8, 1, 384]);  mul_208 = None
    view_358: "f32[8, 384]" = torch.ops.aten.view.default(view_357, [8, 384]);  view_357 = None
    permute_201: "f32[384, 8]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_37: "f32[384, 384]" = torch.ops.aten.mm.default(permute_201, view_319);  permute_201 = view_319 = None
    permute_202: "f32[384, 384]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    permute_203: "f32[384, 384]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_38: "f32[8, 384]" = torch.ops.aten.mm.default(view_358, permute_203);  view_358 = permute_203 = None
    view_359: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_38, [8, 1, 384]);  mm_38 = None
    permute_204: "f32[384, 384]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    full_11: "f32[8, 1, 384]" = torch.ops.aten.full.default([8, 1, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[8, 1, 384]" = torch.ops.aten.slice_scatter.default(full_11, view_359, 2, 0, 9223372036854775807);  full_11 = view_359 = None
    full_12: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_6: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_12, slice_scatter_5, 1, 0, 1);  full_12 = slice_scatter_5 = None
    full_13: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_7: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_13, slice_scatter_6, 0, 0, 9223372036854775807);  full_13 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    cat_3: "f32[16, 12, 197, 32]" = torch.ops.aten.cat.default([permute_200, view_352]);  permute_200 = view_352 = None
    view_360: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.view.default(cat_3, [2, 8, 12, 197, 32]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_205: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.permute.default(view_360, [1, 3, 0, 2, 4]);  view_360 = None
    clone_199: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_361: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_199, [8, 197, 768]);  clone_199 = None
    view_362: "f32[1576, 768]" = torch.ops.aten.view.default(view_361, [1576, 768]);  view_361 = None
    permute_206: "f32[768, 1576]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_39: "f32[768, 384]" = torch.ops.aten.mm.default(permute_206, view_316);  permute_206 = view_316 = None
    permute_207: "f32[384, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    permute_208: "f32[768, 384]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_40: "f32[1576, 384]" = torch.ops.aten.mm.default(view_362, permute_208);  view_362 = permute_208 = None
    view_363: "f32[8, 197, 384]" = torch.ops.aten.view.default(mm_40, [8, 197, 384]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_184: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_7, view_363);  slice_scatter_7 = view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_209: "f32[768, 384]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    sub_71: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_127);  cat_1 = getitem_127 = None
    mul_209: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_41);  sub_71 = None
    mul_210: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_184, primals_234);  primals_234 = None
    mul_211: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_210, 384)
    sum_35: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True)
    mul_212: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_210, mul_209);  mul_210 = None
    sum_36: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
    mul_213: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_209, sum_36);  sum_36 = None
    sub_72: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_211, sum_35);  mul_211 = sum_35 = None
    sub_73: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_72, mul_213);  sub_72 = mul_213 = None
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 384);  rsqrt_41 = None
    mul_214: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_22, sub_73);  div_22 = sub_73 = None
    mul_215: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_184, mul_209);  mul_209 = None
    sum_37: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
    sum_38: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_184, [0, 1]);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_185: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_4, mul_214);  slice_scatter_4 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    full_14: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_8: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_14, add_183, 1, 0, 1);  full_14 = add_183 = None
    full_15: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_9: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_15, slice_scatter_8, 0, 0, 9223372036854775807);  full_15 = slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    add_186: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_185, slice_scatter_9);  add_185 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_28: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_186, 1, 0, 1)
    slice_29: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_186, 1, 1, 197);  add_186 = None
    full_16: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_10: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_16, slice_29, 1, 1, 9223372036854775807);  full_16 = slice_29 = None
    full_17: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_11: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_17, slice_scatter_10, 0, 0, 9223372036854775807);  full_17 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_364: "f32[8, 384]" = torch.ops.aten.view.default(slice_28, [8, 384])
    permute_210: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    mm_41: "f32[8, 1152]" = torch.ops.aten.mm.default(view_364, permute_210);  permute_210 = None
    permute_211: "f32[384, 8]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_42: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_211, view_314);  permute_211 = view_314 = None
    permute_212: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_39: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[384]" = torch.ops.aten.view.default(sum_39, [384]);  sum_39 = None
    permute_213: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    view_366: "f32[8, 1, 1152]" = torch.ops.aten.view.default(mm_41, [8, 1, 1152]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_216: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476)
    erf_21: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_187: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_217: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(add_187, 0.5);  add_187 = None
    mul_218: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, view_313)
    mul_219: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_21: "f32[8, 1, 1152]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_221: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, mul_220);  view_313 = mul_220 = None
    add_188: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_366, add_188);  view_366 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_367: "f32[8, 1152]" = torch.ops.aten.view.default(mul_222, [8, 1152]);  mul_222 = None
    permute_214: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    mm_43: "f32[8, 384]" = torch.ops.aten.mm.default(view_367, permute_214);  permute_214 = None
    permute_215: "f32[1152, 8]" = torch.ops.aten.permute.default(view_367, [1, 0])
    mm_44: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_215, view_312);  permute_215 = view_312 = None
    permute_216: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_40: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
    view_368: "f32[1152]" = torch.ops.aten.view.default(sum_40, [1152]);  sum_40 = None
    permute_217: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_369: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_43, [8, 1, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    sub_74: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_164, getitem_125);  add_164 = getitem_125 = None
    mul_223: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_40);  sub_74 = None
    mul_224: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_369, primals_228);  primals_228 = None
    mul_225: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_224, 384)
    sum_41: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_224, mul_223);  mul_224 = None
    sum_42: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_223, sum_42);  sum_42 = None
    sub_75: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(mul_225, sum_41);  mul_225 = sum_41 = None
    sub_76: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(sub_75, mul_227);  sub_75 = mul_227 = None
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 384);  rsqrt_40 = None
    mul_228: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(div_23, sub_76);  div_23 = sub_76 = None
    mul_229: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_369, mul_223);  mul_223 = None
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_369, [0, 1]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_189: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_28, mul_228);  slice_28 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_370: "f32[8, 384]" = torch.ops.aten.view.default(add_189, [8, 384])
    permute_218: "f32[384, 384]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    mm_45: "f32[8, 384]" = torch.ops.aten.mm.default(view_370, permute_218);  permute_218 = None
    permute_219: "f32[384, 8]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_46: "f32[384, 384]" = torch.ops.aten.mm.default(permute_219, view_310);  permute_219 = view_310 = None
    permute_220: "f32[384, 384]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_45: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[384]" = torch.ops.aten.view.default(sum_45, [384]);  sum_45 = None
    permute_221: "f32[384, 384]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_372: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_45, [8, 1, 384]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    view_373: "f32[8, 1, 12, 32]" = torch.ops.aten.view.default(view_372, [8, 1, 12, 32]);  view_372 = None
    permute_222: "f32[8, 12, 1, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    view_374: "f32[96, 1, 32]" = torch.ops.aten.view.default(permute_222, [96, 1, 32]);  permute_222 = None
    permute_223: "f32[96, 197, 1]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
    bmm_40: "f32[96, 197, 32]" = torch.ops.aten.bmm.default(permute_223, view_374);  permute_223 = None
    permute_224: "f32[96, 32, 197]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    bmm_41: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_374, permute_224);  view_374 = permute_224 = None
    view_375: "f32[8, 12, 197, 32]" = torch.ops.aten.view.default(bmm_40, [8, 12, 197, 32]);  bmm_40 = None
    view_376: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_41, [8, 12, 1, 197]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_230: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(view_376, alias_24);  view_376 = None
    sum_46: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [-1], True)
    mul_231: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(alias_24, sum_46);  alias_24 = sum_46 = None
    sub_77: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    view_377: "f32[96, 1, 197]" = torch.ops.aten.view.default(sub_77, [96, 1, 197]);  sub_77 = None
    permute_225: "f32[96, 32, 1]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    bmm_42: "f32[96, 32, 197]" = torch.ops.aten.bmm.default(permute_225, view_377);  permute_225 = None
    permute_226: "f32[96, 197, 32]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    bmm_43: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_377, permute_226);  view_377 = permute_226 = None
    view_378: "f32[8, 12, 32, 197]" = torch.ops.aten.view.default(bmm_42, [8, 12, 32, 197]);  bmm_42 = None
    view_379: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_43, [8, 12, 1, 32]);  bmm_43 = None
    permute_227: "f32[8, 12, 197, 32]" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2]);  view_378 = None
    mul_232: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_379, 0.1767766952966369);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    view_380: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_232, [8, 1, 384]);  mul_232 = None
    view_381: "f32[8, 384]" = torch.ops.aten.view.default(view_380, [8, 384]);  view_380 = None
    permute_228: "f32[384, 8]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_47: "f32[384, 384]" = torch.ops.aten.mm.default(permute_228, view_300);  permute_228 = view_300 = None
    permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    permute_230: "f32[384, 384]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    mm_48: "f32[8, 384]" = torch.ops.aten.mm.default(view_381, permute_230);  view_381 = permute_230 = None
    view_382: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_48, [8, 1, 384]);  mm_48 = None
    permute_231: "f32[384, 384]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    full_18: "f32[8, 1, 384]" = torch.ops.aten.full.default([8, 1, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_12: "f32[8, 1, 384]" = torch.ops.aten.slice_scatter.default(full_18, view_382, 2, 0, 9223372036854775807);  full_18 = view_382 = None
    full_19: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_13: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_19, slice_scatter_12, 1, 0, 1);  full_19 = slice_scatter_12 = None
    full_20: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_14: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_20, slice_scatter_13, 0, 0, 9223372036854775807);  full_20 = slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    cat_4: "f32[16, 12, 197, 32]" = torch.ops.aten.cat.default([permute_227, view_375]);  permute_227 = view_375 = None
    view_383: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.view.default(cat_4, [2, 8, 12, 197, 32]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_232: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.permute.default(view_383, [1, 3, 0, 2, 4]);  view_383 = None
    clone_200: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    view_384: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_200, [8, 197, 768]);  clone_200 = None
    view_385: "f32[1576, 768]" = torch.ops.aten.view.default(view_384, [1576, 768]);  view_384 = None
    permute_233: "f32[768, 1576]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_49: "f32[768, 384]" = torch.ops.aten.mm.default(permute_233, view_297);  permute_233 = view_297 = None
    permute_234: "f32[384, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    permute_235: "f32[768, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    mm_50: "f32[1576, 384]" = torch.ops.aten.mm.default(view_385, permute_235);  view_385 = permute_235 = None
    view_386: "f32[8, 197, 384]" = torch.ops.aten.view.default(mm_50, [8, 197, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_190: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_14, view_386);  slice_scatter_14 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_236: "f32[768, 384]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    sub_78: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat, getitem_121);  cat = getitem_121 = None
    mul_233: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_39);  sub_78 = None
    mul_234: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_190, primals_222);  primals_222 = None
    mul_235: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_234, 384)
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_234, mul_233);  mul_234 = None
    sum_48: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_233, sum_48);  sum_48 = None
    sub_79: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_235, sum_47);  mul_235 = sum_47 = None
    sub_80: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_79, mul_237);  sub_79 = mul_237 = None
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 384);  rsqrt_39 = None
    mul_238: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_24, sub_80);  div_24 = sub_80 = None
    mul_239: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_190, mul_233);  mul_233 = None
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_50: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_191: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_11, mul_238);  slice_scatter_11 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    full_21: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_15: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_21, add_189, 1, 0, 1);  full_21 = add_189 = None
    full_22: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_16: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_22, slice_scatter_15, 0, 0, 9223372036854775807);  full_22 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    add_192: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_191, slice_scatter_16);  add_191 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    slice_30: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_192, 1, 0, 1)
    slice_31: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_192, 1, 1, 197);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    sum_51: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(slice_30, [0], True);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    view_387: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(slice_31, [8, 14, 14, 384]);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_201: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_387, memory_format = torch.contiguous_format)
    view_388: "f32[1568, 384]" = torch.ops.aten.view.default(clone_201, [1568, 384]);  clone_201 = None
    permute_237: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    mm_51: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_388, permute_237);  permute_237 = None
    permute_238: "f32[384, 1568]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_52: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_238, view_294);  permute_238 = view_294 = None
    permute_239: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_52: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[384]" = torch.ops.aten.view.default(sum_52, [384]);  sum_52 = None
    permute_240: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_390: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_51, [8, 14, 14, 1152]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_240: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.7071067811865476)
    erf_22: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_193: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_241: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
    mul_242: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, view_293)
    mul_243: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_22: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, mul_244);  view_293 = mul_244 = None
    add_194: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_390, add_194);  view_390 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_246, [1568, 1152]);  mul_246 = None
    permute_241: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    mm_53: "f32[1568, 384]" = torch.ops.aten.mm.default(view_391, permute_241);  permute_241 = None
    permute_242: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_54: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_242, view_292);  permute_242 = view_292 = None
    permute_243: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_53: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[1152]" = torch.ops.aten.view.default(sum_53, [1152]);  sum_53 = None
    permute_244: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_393: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_53, [8, 14, 14, 384]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_202: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format);  add_157 = None
    sub_81: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_202, getitem_119);  clone_202 = getitem_119 = None
    mul_247: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_38);  sub_81 = None
    mul_248: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_393, primals_216);  primals_216 = None
    mul_249: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_248, 384)
    sum_54: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [3], True)
    mul_250: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_248, mul_247);  mul_248 = None
    sum_55: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [3], True);  mul_250 = None
    mul_251: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_247, sum_55);  sum_55 = None
    sub_82: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_249, sum_54);  mul_249 = sum_54 = None
    sub_83: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_82, mul_251);  sub_82 = mul_251 = None
    div_25: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 384);  rsqrt_38 = None
    mul_252: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_25, sub_83);  div_25 = sub_83 = None
    mul_253: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_393, mul_247);  mul_247 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1, 2]);  mul_253 = None
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_393, [0, 1, 2]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_195: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(view_387, mul_252);  view_387 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_394: "f32[1568, 384]" = torch.ops.aten.view.default(add_195, [1568, 384])
    permute_245: "f32[384, 384]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_55: "f32[1568, 384]" = torch.ops.aten.mm.default(view_394, permute_245);  permute_245 = None
    permute_246: "f32[384, 1568]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_56: "f32[384, 384]" = torch.ops.aten.mm.default(permute_246, view_290);  permute_246 = view_290 = None
    permute_247: "f32[384, 384]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_58: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[384]" = torch.ops.aten.view.default(sum_58, [384]);  sum_58 = None
    permute_248: "f32[384, 384]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_396: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_55, [8, 14, 14, 384]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_397: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_396, [8, 196, 12, 32]);  view_396 = None
    permute_249: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    clone_203: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_398: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_203, [96, 196, 32]);  clone_203 = None
    permute_250: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    bmm_44: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_250, view_398);  permute_250 = None
    permute_251: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    bmm_45: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_398, permute_251);  view_398 = permute_251 = None
    view_399: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_44, [8, 12, 196, 32]);  bmm_44 = None
    view_400: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_45, [8, 12, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_254: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_400, alias_25);  view_400 = None
    sum_59: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [-1], True)
    mul_255: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_59);  alias_25 = sum_59 = None
    sub_84: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_256: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_84, 0.1767766952966369);  sub_84 = None
    view_401: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_256, [96, 196, 196]);  mul_256 = None
    permute_252: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    bmm_46: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_252, view_401);  permute_252 = None
    permute_253: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    bmm_47: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_401, permute_253);  view_401 = permute_253 = None
    view_402: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_46, [8, 12, 32, 196]);  bmm_46 = None
    view_403: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_47, [8, 12, 196, 32]);  bmm_47 = None
    permute_254: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_403, permute_254, view_399]);  view_403 = permute_254 = view_399 = None
    view_404: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_5, [3, 8, 12, 196, 32]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_255: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_404, [1, 3, 0, 2, 4]);  view_404 = None
    clone_204: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_405: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_204, [8, 14, 14, 1152]);  clone_204 = None
    view_406: "f32[1568, 1152]" = torch.ops.aten.view.default(view_405, [1568, 1152]);  view_405 = None
    permute_256: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_57: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_256, view_280);  permute_256 = view_280 = None
    permute_257: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    permute_258: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_58: "f32[1568, 384]" = torch.ops.aten.mm.default(view_406, permute_258);  view_406 = permute_258 = None
    view_407: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_58, [8, 14, 14, 384]);  mm_58 = None
    permute_259: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_205: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_154, memory_format = torch.contiguous_format);  add_154 = None
    sub_85: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_205, getitem_114);  clone_205 = getitem_114 = None
    mul_257: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_37);  sub_85 = None
    mul_258: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_407, primals_211);  primals_211 = None
    mul_259: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_258, 384)
    sum_60: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [3], True)
    mul_260: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_258, mul_257);  mul_258 = None
    sum_61: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [3], True);  mul_260 = None
    mul_261: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_257, sum_61);  sum_61 = None
    sub_86: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_259, sum_60);  mul_259 = sum_60 = None
    sub_87: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_86, mul_261);  sub_86 = mul_261 = None
    div_26: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 384);  rsqrt_37 = None
    mul_262: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_26, sub_87);  div_26 = sub_87 = None
    mul_263: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_407, mul_257);  mul_257 = None
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1, 2]);  mul_263 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_407, [0, 1, 2]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_196: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_195, mul_262);  add_195 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_408: "f32[1568, 384]" = torch.ops.aten.view.default(add_196, [1568, 384])
    permute_260: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    mm_59: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_408, permute_260);  permute_260 = None
    permute_261: "f32[384, 1568]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_60: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_261, view_278);  permute_261 = view_278 = None
    permute_262: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_64: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[384]" = torch.ops.aten.view.default(sum_64, [384]);  sum_64 = None
    permute_263: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_410: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_59, [8, 14, 14, 1152]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_264: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476)
    erf_23: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
    add_197: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_265: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_197, 0.5);  add_197 = None
    mul_266: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, view_277)
    mul_267: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_266, -0.5);  mul_266 = None
    exp_23: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_267);  mul_267 = None
    mul_268: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_269: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, mul_268);  view_277 = mul_268 = None
    add_198: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_265, mul_269);  mul_265 = mul_269 = None
    mul_270: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_410, add_198);  view_410 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_411: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_270, [1568, 1152]);  mul_270 = None
    permute_264: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    mm_61: "f32[1568, 384]" = torch.ops.aten.mm.default(view_411, permute_264);  permute_264 = None
    permute_265: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_62: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_265, view_276);  permute_265 = view_276 = None
    permute_266: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_65: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[1152]" = torch.ops.aten.view.default(sum_65, [1152]);  sum_65 = None
    permute_267: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_413: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_61, [8, 14, 14, 384]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_206: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format);  add_150 = None
    sub_88: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_206, getitem_112);  clone_206 = getitem_112 = None
    mul_271: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_36);  sub_88 = None
    mul_272: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_413, primals_205);  primals_205 = None
    mul_273: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_272, 384)
    sum_66: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [3], True)
    mul_274: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_272, mul_271);  mul_272 = None
    sum_67: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [3], True);  mul_274 = None
    mul_275: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_271, sum_67);  sum_67 = None
    sub_89: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_273, sum_66);  mul_273 = sum_66 = None
    sub_90: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_89, mul_275);  sub_89 = mul_275 = None
    div_27: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 384);  rsqrt_36 = None
    mul_276: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_27, sub_90);  div_27 = sub_90 = None
    mul_277: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_413, mul_271);  mul_271 = None
    sum_68: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1, 2]);  mul_277 = None
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_413, [0, 1, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_199: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_196, mul_276);  add_196 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_414: "f32[1568, 384]" = torch.ops.aten.view.default(add_199, [1568, 384])
    permute_268: "f32[384, 384]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    mm_63: "f32[1568, 384]" = torch.ops.aten.mm.default(view_414, permute_268);  permute_268 = None
    permute_269: "f32[384, 1568]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_64: "f32[384, 384]" = torch.ops.aten.mm.default(permute_269, view_274);  permute_269 = view_274 = None
    permute_270: "f32[384, 384]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_70: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[384]" = torch.ops.aten.view.default(sum_70, [384]);  sum_70 = None
    permute_271: "f32[384, 384]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_416: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_63, [8, 14, 14, 384]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_417: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_416, [8, 196, 12, 32]);  view_416 = None
    permute_272: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    clone_207: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_418: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_207, [96, 196, 32]);  clone_207 = None
    permute_273: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    bmm_48: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_273, view_418);  permute_273 = None
    permute_274: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    bmm_49: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_418, permute_274);  view_418 = permute_274 = None
    view_419: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_48, [8, 12, 196, 32]);  bmm_48 = None
    view_420: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_49, [8, 12, 196, 196]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_26: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_278: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_420, alias_26);  view_420 = None
    sum_71: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [-1], True)
    mul_279: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_71);  alias_26 = sum_71 = None
    sub_91: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_280: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_91, 0.1767766952966369);  sub_91 = None
    view_421: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_280, [96, 196, 196]);  mul_280 = None
    permute_275: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    bmm_50: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_275, view_421);  permute_275 = None
    permute_276: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    bmm_51: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_421, permute_276);  view_421 = permute_276 = None
    view_422: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_50, [8, 12, 32, 196]);  bmm_50 = None
    view_423: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_51, [8, 12, 196, 32]);  bmm_51 = None
    permute_277: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_422, [0, 1, 3, 2]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_423, permute_277, view_419]);  view_423 = permute_277 = view_419 = None
    view_424: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_6, [3, 8, 12, 196, 32]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_278: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_424, [1, 3, 0, 2, 4]);  view_424 = None
    clone_208: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
    view_425: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_208, [8, 14, 14, 1152]);  clone_208 = None
    view_426: "f32[1568, 1152]" = torch.ops.aten.view.default(view_425, [1568, 1152]);  view_425 = None
    permute_279: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_65: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_279, view_264);  permute_279 = view_264 = None
    permute_280: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    permute_281: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_66: "f32[1568, 384]" = torch.ops.aten.mm.default(view_426, permute_281);  view_426 = permute_281 = None
    view_427: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_66, [8, 14, 14, 384]);  mm_66 = None
    permute_282: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_209: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_147, memory_format = torch.contiguous_format);  add_147 = None
    sub_92: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_209, getitem_107);  clone_209 = getitem_107 = None
    mul_281: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_35);  sub_92 = None
    mul_282: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_427, primals_200);  primals_200 = None
    mul_283: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_282, 384)
    sum_72: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [3], True)
    mul_284: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_282, mul_281);  mul_282 = None
    sum_73: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True);  mul_284 = None
    mul_285: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_281, sum_73);  sum_73 = None
    sub_93: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_283, sum_72);  mul_283 = sum_72 = None
    sub_94: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_93, mul_285);  sub_93 = mul_285 = None
    div_28: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 384);  rsqrt_35 = None
    mul_286: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_28, sub_94);  div_28 = sub_94 = None
    mul_287: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_427, mul_281);  mul_281 = None
    sum_74: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1, 2]);  mul_287 = None
    sum_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_200: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_199, mul_286);  add_199 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_428: "f32[1568, 384]" = torch.ops.aten.view.default(add_200, [1568, 384])
    permute_283: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_67: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_428, permute_283);  permute_283 = None
    permute_284: "f32[384, 1568]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_68: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_284, view_262);  permute_284 = view_262 = None
    permute_285: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_76: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[384]" = torch.ops.aten.view.default(sum_76, [384]);  sum_76 = None
    permute_286: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_430: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_67, [8, 14, 14, 1152]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_288: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_24: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_288);  mul_288 = None
    add_201: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_289: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_201, 0.5);  add_201 = None
    mul_290: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_291: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_24: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, mul_292);  view_261 = mul_292 = None
    add_202: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_430, add_202);  view_430 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_431: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_294, [1568, 1152]);  mul_294 = None
    permute_287: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_69: "f32[1568, 384]" = torch.ops.aten.mm.default(view_431, permute_287);  permute_287 = None
    permute_288: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_70: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_288, view_260);  permute_288 = view_260 = None
    permute_289: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_77: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[1152]" = torch.ops.aten.view.default(sum_77, [1152]);  sum_77 = None
    permute_290: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_433: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_69, [8, 14, 14, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_210: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format);  add_143 = None
    sub_95: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_210, getitem_105);  clone_210 = getitem_105 = None
    mul_295: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_34);  sub_95 = None
    mul_296: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_433, primals_194);  primals_194 = None
    mul_297: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_296, 384)
    sum_78: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [3], True)
    mul_298: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_296, mul_295);  mul_296 = None
    sum_79: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True);  mul_298 = None
    mul_299: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_295, sum_79);  sum_79 = None
    sub_96: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_297, sum_78);  mul_297 = sum_78 = None
    sub_97: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_96, mul_299);  sub_96 = mul_299 = None
    div_29: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 384);  rsqrt_34 = None
    mul_300: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_29, sub_97);  div_29 = sub_97 = None
    mul_301: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_433, mul_295);  mul_295 = None
    sum_80: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1, 2]);  mul_301 = None
    sum_81: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1, 2]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_203: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_200, mul_300);  add_200 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_434: "f32[1568, 384]" = torch.ops.aten.view.default(add_203, [1568, 384])
    permute_291: "f32[384, 384]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_71: "f32[1568, 384]" = torch.ops.aten.mm.default(view_434, permute_291);  permute_291 = None
    permute_292: "f32[384, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_72: "f32[384, 384]" = torch.ops.aten.mm.default(permute_292, view_258);  permute_292 = view_258 = None
    permute_293: "f32[384, 384]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_82: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[384]" = torch.ops.aten.view.default(sum_82, [384]);  sum_82 = None
    permute_294: "f32[384, 384]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_436: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_71, [8, 14, 14, 384]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_437: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_436, [8, 196, 12, 32]);  view_436 = None
    permute_295: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    clone_211: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_438: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_211, [96, 196, 32]);  clone_211 = None
    permute_296: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_52: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_296, view_438);  permute_296 = None
    permute_297: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    bmm_53: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_438, permute_297);  view_438 = permute_297 = None
    view_439: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_52, [8, 12, 196, 32]);  bmm_52 = None
    view_440: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_53, [8, 12, 196, 196]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_302: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_440, alias_27);  view_440 = None
    sum_83: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [-1], True)
    mul_303: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_83);  alias_27 = sum_83 = None
    sub_98: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_304: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_98, 0.1767766952966369);  sub_98 = None
    view_441: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_304, [96, 196, 196]);  mul_304 = None
    permute_298: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_54: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_298, view_441);  permute_298 = None
    permute_299: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_55: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_441, permute_299);  view_441 = permute_299 = None
    view_442: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_54, [8, 12, 32, 196]);  bmm_54 = None
    view_443: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_55, [8, 12, 196, 32]);  bmm_55 = None
    permute_300: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_442, [0, 1, 3, 2]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_443, permute_300, view_439]);  view_443 = permute_300 = view_439 = None
    view_444: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_7, [3, 8, 12, 196, 32]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_301: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_444, [1, 3, 0, 2, 4]);  view_444 = None
    clone_212: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_301, memory_format = torch.contiguous_format);  permute_301 = None
    view_445: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_212, [8, 14, 14, 1152]);  clone_212 = None
    view_446: "f32[1568, 1152]" = torch.ops.aten.view.default(view_445, [1568, 1152]);  view_445 = None
    permute_302: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_73: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_302, view_248);  permute_302 = view_248 = None
    permute_303: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    permute_304: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_74: "f32[1568, 384]" = torch.ops.aten.mm.default(view_446, permute_304);  view_446 = permute_304 = None
    view_447: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_74, [8, 14, 14, 384]);  mm_74 = None
    permute_305: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_213: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format);  add_140 = None
    sub_99: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_213, getitem_100);  clone_213 = getitem_100 = None
    mul_305: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_33);  sub_99 = None
    mul_306: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_447, primals_189);  primals_189 = None
    mul_307: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_306, 384)
    sum_84: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [3], True)
    mul_308: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_306, mul_305);  mul_306 = None
    sum_85: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [3], True);  mul_308 = None
    mul_309: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_305, sum_85);  sum_85 = None
    sub_100: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_307, sum_84);  mul_307 = sum_84 = None
    sub_101: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_100, mul_309);  sub_100 = mul_309 = None
    div_30: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 384);  rsqrt_33 = None
    mul_310: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_30, sub_101);  div_30 = sub_101 = None
    mul_311: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_447, mul_305);  mul_305 = None
    sum_86: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1, 2]);  mul_311 = None
    sum_87: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_447, [0, 1, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_204: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_203, mul_310);  add_203 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[1568, 384]" = torch.ops.aten.view.default(add_204, [1568, 384])
    permute_306: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_75: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_448, permute_306);  permute_306 = None
    permute_307: "f32[384, 1568]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_76: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_307, view_246);  permute_307 = view_246 = None
    permute_308: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_88: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[384]" = torch.ops.aten.view.default(sum_88, [384]);  sum_88 = None
    permute_309: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_450: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_75, [8, 14, 14, 1152]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_312: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476)
    erf_25: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_312);  mul_312 = None
    add_205: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_313: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_205, 0.5);  add_205 = None
    mul_314: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, view_245)
    mul_315: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_314, -0.5);  mul_314 = None
    exp_25: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_315);  mul_315 = None
    mul_316: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_317: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, mul_316);  view_245 = mul_316 = None
    add_206: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_313, mul_317);  mul_313 = mul_317 = None
    mul_318: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_450, add_206);  view_450 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_451: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_318, [1568, 1152]);  mul_318 = None
    permute_310: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_77: "f32[1568, 384]" = torch.ops.aten.mm.default(view_451, permute_310);  permute_310 = None
    permute_311: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_78: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_311, view_244);  permute_311 = view_244 = None
    permute_312: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_89: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[1152]" = torch.ops.aten.view.default(sum_89, [1152]);  sum_89 = None
    permute_313: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_453: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_77, [8, 14, 14, 384]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_214: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format);  add_136 = None
    sub_102: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_214, getitem_98);  clone_214 = getitem_98 = None
    mul_319: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_32);  sub_102 = None
    mul_320: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_453, primals_183);  primals_183 = None
    mul_321: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_320, 384)
    sum_90: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [3], True)
    mul_322: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_320, mul_319);  mul_320 = None
    sum_91: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [3], True);  mul_322 = None
    mul_323: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_319, sum_91);  sum_91 = None
    sub_103: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_321, sum_90);  mul_321 = sum_90 = None
    sub_104: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_103, mul_323);  sub_103 = mul_323 = None
    div_31: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 384);  rsqrt_32 = None
    mul_324: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_31, sub_104);  div_31 = sub_104 = None
    mul_325: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_453, mul_319);  mul_319 = None
    sum_92: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1, 2]);  mul_325 = None
    sum_93: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_453, [0, 1, 2]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_207: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_204, mul_324);  add_204 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_454: "f32[1568, 384]" = torch.ops.aten.view.default(add_207, [1568, 384])
    permute_314: "f32[384, 384]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_79: "f32[1568, 384]" = torch.ops.aten.mm.default(view_454, permute_314);  permute_314 = None
    permute_315: "f32[384, 1568]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_80: "f32[384, 384]" = torch.ops.aten.mm.default(permute_315, view_242);  permute_315 = view_242 = None
    permute_316: "f32[384, 384]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_94: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[384]" = torch.ops.aten.view.default(sum_94, [384]);  sum_94 = None
    permute_317: "f32[384, 384]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_456: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_79, [8, 14, 14, 384]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_457: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_456, [8, 196, 12, 32]);  view_456 = None
    permute_318: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    clone_215: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    view_458: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_215, [96, 196, 32]);  clone_215 = None
    permute_319: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
    bmm_56: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_319, view_458);  permute_319 = None
    permute_320: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    bmm_57: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_458, permute_320);  view_458 = permute_320 = None
    view_459: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_56, [8, 12, 196, 32]);  bmm_56 = None
    view_460: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_57, [8, 12, 196, 196]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_28: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_326: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_460, alias_28);  view_460 = None
    sum_95: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
    mul_327: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_28, sum_95);  alias_28 = sum_95 = None
    sub_105: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_328: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_105, 0.1767766952966369);  sub_105 = None
    view_461: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_328, [96, 196, 196]);  mul_328 = None
    permute_321: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_58: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_321, view_461);  permute_321 = None
    permute_322: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
    bmm_59: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_461, permute_322);  view_461 = permute_322 = None
    view_462: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_58, [8, 12, 32, 196]);  bmm_58 = None
    view_463: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_59, [8, 12, 196, 32]);  bmm_59 = None
    permute_323: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_462, [0, 1, 3, 2]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_463, permute_323, view_459]);  view_463 = permute_323 = view_459 = None
    view_464: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_8, [3, 8, 12, 196, 32]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_324: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_464, [1, 3, 0, 2, 4]);  view_464 = None
    clone_216: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
    view_465: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_216, [8, 14, 14, 1152]);  clone_216 = None
    view_466: "f32[1568, 1152]" = torch.ops.aten.view.default(view_465, [1568, 1152]);  view_465 = None
    permute_325: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_81: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_325, view_232);  permute_325 = view_232 = None
    permute_326: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    permute_327: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_82: "f32[1568, 384]" = torch.ops.aten.mm.default(view_466, permute_327);  view_466 = permute_327 = None
    view_467: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_82, [8, 14, 14, 384]);  mm_82 = None
    permute_328: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_217: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format);  add_133 = None
    sub_106: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_217, getitem_93);  clone_217 = getitem_93 = None
    mul_329: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_31);  sub_106 = None
    mul_330: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_467, primals_178);  primals_178 = None
    mul_331: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_330, 384)
    sum_96: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [3], True)
    mul_332: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_330, mul_329);  mul_330 = None
    sum_97: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [3], True);  mul_332 = None
    mul_333: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_329, sum_97);  sum_97 = None
    sub_107: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_331, sum_96);  mul_331 = sum_96 = None
    sub_108: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_107, mul_333);  sub_107 = mul_333 = None
    div_32: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 384);  rsqrt_31 = None
    mul_334: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_32, sub_108);  div_32 = sub_108 = None
    mul_335: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_467, mul_329);  mul_329 = None
    sum_98: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1, 2]);  mul_335 = None
    sum_99: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_467, [0, 1, 2]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_208: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_207, mul_334);  add_207 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_468: "f32[1568, 384]" = torch.ops.aten.view.default(add_208, [1568, 384])
    permute_329: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_83: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_468, permute_329);  permute_329 = None
    permute_330: "f32[384, 1568]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_84: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_330, view_230);  permute_330 = view_230 = None
    permute_331: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_100: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
    view_469: "f32[384]" = torch.ops.aten.view.default(sum_100, [384]);  sum_100 = None
    permute_332: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_470: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_83, [8, 14, 14, 1152]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_336: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476)
    erf_26: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_336);  mul_336 = None
    add_209: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_337: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_209, 0.5);  add_209 = None
    mul_338: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, view_229)
    mul_339: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_338, -0.5);  mul_338 = None
    exp_26: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_339);  mul_339 = None
    mul_340: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_341: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, mul_340);  view_229 = mul_340 = None
    add_210: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_337, mul_341);  mul_337 = mul_341 = None
    mul_342: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_470, add_210);  view_470 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_471: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_342, [1568, 1152]);  mul_342 = None
    permute_333: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    mm_85: "f32[1568, 384]" = torch.ops.aten.mm.default(view_471, permute_333);  permute_333 = None
    permute_334: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_86: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_334, view_228);  permute_334 = view_228 = None
    permute_335: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_101: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[1152]" = torch.ops.aten.view.default(sum_101, [1152]);  sum_101 = None
    permute_336: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_473: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_85, [8, 14, 14, 384]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_218: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format);  add_129 = None
    sub_109: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_218, getitem_91);  clone_218 = getitem_91 = None
    mul_343: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_30);  sub_109 = None
    mul_344: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_473, primals_172);  primals_172 = None
    mul_345: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_344, 384)
    sum_102: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [3], True)
    mul_346: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_344, mul_343);  mul_344 = None
    sum_103: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [3], True);  mul_346 = None
    mul_347: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_343, sum_103);  sum_103 = None
    sub_110: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_345, sum_102);  mul_345 = sum_102 = None
    sub_111: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_110, mul_347);  sub_110 = mul_347 = None
    div_33: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 384);  rsqrt_30 = None
    mul_348: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_33, sub_111);  div_33 = sub_111 = None
    mul_349: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_473, mul_343);  mul_343 = None
    sum_104: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1, 2]);  mul_349 = None
    sum_105: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1, 2]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_211: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_208, mul_348);  add_208 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_474: "f32[1568, 384]" = torch.ops.aten.view.default(add_211, [1568, 384])
    permute_337: "f32[384, 384]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_87: "f32[1568, 384]" = torch.ops.aten.mm.default(view_474, permute_337);  permute_337 = None
    permute_338: "f32[384, 1568]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_88: "f32[384, 384]" = torch.ops.aten.mm.default(permute_338, view_226);  permute_338 = view_226 = None
    permute_339: "f32[384, 384]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_106: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[384]" = torch.ops.aten.view.default(sum_106, [384]);  sum_106 = None
    permute_340: "f32[384, 384]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_476: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_87, [8, 14, 14, 384]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_477: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_476, [8, 196, 12, 32]);  view_476 = None
    permute_341: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
    clone_219: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None
    view_478: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_219, [96, 196, 32]);  clone_219 = None
    permute_342: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
    bmm_60: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_342, view_478);  permute_342 = None
    permute_343: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    bmm_61: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_478, permute_343);  view_478 = permute_343 = None
    view_479: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_60, [8, 12, 196, 32]);  bmm_60 = None
    view_480: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_61, [8, 12, 196, 196]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_29: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_350: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_480, alias_29);  view_480 = None
    sum_107: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [-1], True)
    mul_351: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_29, sum_107);  alias_29 = sum_107 = None
    sub_112: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_352: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_112, 0.1767766952966369);  sub_112 = None
    view_481: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_352, [96, 196, 196]);  mul_352 = None
    permute_344: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    bmm_62: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_344, view_481);  permute_344 = None
    permute_345: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    bmm_63: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_481, permute_345);  view_481 = permute_345 = None
    view_482: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_62, [8, 12, 32, 196]);  bmm_62 = None
    view_483: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_63, [8, 12, 196, 32]);  bmm_63 = None
    permute_346: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_482, [0, 1, 3, 2]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_483, permute_346, view_479]);  view_483 = permute_346 = view_479 = None
    view_484: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_9, [3, 8, 12, 196, 32]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_347: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_484, [1, 3, 0, 2, 4]);  view_484 = None
    clone_220: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    view_485: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_220, [8, 14, 14, 1152]);  clone_220 = None
    view_486: "f32[1568, 1152]" = torch.ops.aten.view.default(view_485, [1568, 1152]);  view_485 = None
    permute_348: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_89: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_348, view_216);  permute_348 = view_216 = None
    permute_349: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    permute_350: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_90: "f32[1568, 384]" = torch.ops.aten.mm.default(view_486, permute_350);  view_486 = permute_350 = None
    view_487: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_90, [8, 14, 14, 384]);  mm_90 = None
    permute_351: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_221: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format);  add_126 = None
    sub_113: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_221, getitem_86);  clone_221 = getitem_86 = None
    mul_353: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_29);  sub_113 = None
    mul_354: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_487, primals_167);  primals_167 = None
    mul_355: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_354, 384)
    sum_108: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [3], True)
    mul_356: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_354, mul_353);  mul_354 = None
    sum_109: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [3], True);  mul_356 = None
    mul_357: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_353, sum_109);  sum_109 = None
    sub_114: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_355, sum_108);  mul_355 = sum_108 = None
    sub_115: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_114, mul_357);  sub_114 = mul_357 = None
    div_34: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 384);  rsqrt_29 = None
    mul_358: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_34, sub_115);  div_34 = sub_115 = None
    mul_359: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_487, mul_353);  mul_353 = None
    sum_110: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1, 2]);  mul_359 = None
    sum_111: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_487, [0, 1, 2]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_212: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_211, mul_358);  add_211 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_488: "f32[1568, 384]" = torch.ops.aten.view.default(add_212, [1568, 384])
    permute_352: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_91: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_488, permute_352);  permute_352 = None
    permute_353: "f32[384, 1568]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_92: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_353, view_214);  permute_353 = view_214 = None
    permute_354: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_112: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[384]" = torch.ops.aten.view.default(sum_112, [384]);  sum_112 = None
    permute_355: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_490: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_91, [8, 14, 14, 1152]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_360: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_27: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_360);  mul_360 = None
    add_213: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_361: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_213, 0.5);  add_213 = None
    mul_362: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_363: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
    exp_27: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
    mul_364: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_365: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, mul_364);  view_213 = mul_364 = None
    add_214: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
    mul_366: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_490, add_214);  view_490 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_491: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_366, [1568, 1152]);  mul_366 = None
    permute_356: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_93: "f32[1568, 384]" = torch.ops.aten.mm.default(view_491, permute_356);  permute_356 = None
    permute_357: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_94: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_357, view_212);  permute_357 = view_212 = None
    permute_358: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_113: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[1152]" = torch.ops.aten.view.default(sum_113, [1152]);  sum_113 = None
    permute_359: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_493: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_93, [8, 14, 14, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_222: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format);  add_122 = None
    sub_116: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_222, getitem_84);  clone_222 = getitem_84 = None
    mul_367: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_28);  sub_116 = None
    mul_368: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_493, primals_161);  primals_161 = None
    mul_369: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_368, 384)
    sum_114: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [3], True)
    mul_370: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_368, mul_367);  mul_368 = None
    sum_115: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [3], True);  mul_370 = None
    mul_371: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_367, sum_115);  sum_115 = None
    sub_117: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_369, sum_114);  mul_369 = sum_114 = None
    sub_118: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_117, mul_371);  sub_117 = mul_371 = None
    div_35: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 384);  rsqrt_28 = None
    mul_372: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_35, sub_118);  div_35 = sub_118 = None
    mul_373: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_493, mul_367);  mul_367 = None
    sum_116: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1, 2]);  mul_373 = None
    sum_117: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_493, [0, 1, 2]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_215: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_212, mul_372);  add_212 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_494: "f32[1568, 384]" = torch.ops.aten.view.default(add_215, [1568, 384])
    permute_360: "f32[384, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_95: "f32[1568, 384]" = torch.ops.aten.mm.default(view_494, permute_360);  permute_360 = None
    permute_361: "f32[384, 1568]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_96: "f32[384, 384]" = torch.ops.aten.mm.default(permute_361, view_210);  permute_361 = view_210 = None
    permute_362: "f32[384, 384]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_118: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
    view_495: "f32[384]" = torch.ops.aten.view.default(sum_118, [384]);  sum_118 = None
    permute_363: "f32[384, 384]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_496: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_95, [8, 14, 14, 384]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_497: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_496, [8, 196, 12, 32]);  view_496 = None
    permute_364: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_223: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
    view_498: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_223, [96, 196, 32]);  clone_223 = None
    permute_365: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_64: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_365, view_498);  permute_365 = None
    permute_366: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_65: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_498, permute_366);  view_498 = permute_366 = None
    view_499: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_64, [8, 12, 196, 32]);  bmm_64 = None
    view_500: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_65, [8, 12, 196, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_30: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_374: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_500, alias_30);  view_500 = None
    sum_119: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [-1], True)
    mul_375: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_30, sum_119);  alias_30 = sum_119 = None
    sub_119: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_376: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_119, 0.1767766952966369);  sub_119 = None
    view_501: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_376, [96, 196, 196]);  mul_376 = None
    permute_367: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    bmm_66: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_367, view_501);  permute_367 = None
    permute_368: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_67: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_501, permute_368);  view_501 = permute_368 = None
    view_502: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_66, [8, 12, 32, 196]);  bmm_66 = None
    view_503: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_67, [8, 12, 196, 32]);  bmm_67 = None
    permute_369: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 2]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_503, permute_369, view_499]);  view_503 = permute_369 = view_499 = None
    view_504: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_10, [3, 8, 12, 196, 32]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_370: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_504, [1, 3, 0, 2, 4]);  view_504 = None
    clone_224: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_505: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_224, [8, 14, 14, 1152]);  clone_224 = None
    view_506: "f32[1568, 1152]" = torch.ops.aten.view.default(view_505, [1568, 1152]);  view_505 = None
    permute_371: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_97: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_371, view_200);  permute_371 = view_200 = None
    permute_372: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    permute_373: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_98: "f32[1568, 384]" = torch.ops.aten.mm.default(view_506, permute_373);  view_506 = permute_373 = None
    view_507: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_98, [8, 14, 14, 384]);  mm_98 = None
    permute_374: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_225: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_119, memory_format = torch.contiguous_format);  add_119 = None
    sub_120: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_225, getitem_79);  clone_225 = getitem_79 = None
    mul_377: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_27);  sub_120 = None
    mul_378: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_507, primals_156);  primals_156 = None
    mul_379: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_378, 384)
    sum_120: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [3], True)
    mul_380: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_378, mul_377);  mul_378 = None
    sum_121: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [3], True);  mul_380 = None
    mul_381: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_377, sum_121);  sum_121 = None
    sub_121: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_379, sum_120);  mul_379 = sum_120 = None
    sub_122: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_121, mul_381);  sub_121 = mul_381 = None
    div_36: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 384);  rsqrt_27 = None
    mul_382: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_36, sub_122);  div_36 = sub_122 = None
    mul_383: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_507, mul_377);  mul_377 = None
    sum_122: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1, 2]);  mul_383 = None
    sum_123: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_507, [0, 1, 2]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_216: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_215, mul_382);  add_215 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_508: "f32[1568, 384]" = torch.ops.aten.view.default(add_216, [1568, 384])
    permute_375: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_99: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_508, permute_375);  permute_375 = None
    permute_376: "f32[384, 1568]" = torch.ops.aten.permute.default(view_508, [1, 0])
    mm_100: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_376, view_198);  permute_376 = view_198 = None
    permute_377: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_124: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True);  view_508 = None
    view_509: "f32[384]" = torch.ops.aten.view.default(sum_124, [384]);  sum_124 = None
    permute_378: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_510: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_99, [8, 14, 14, 1152]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_384: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_28: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_384);  mul_384 = None
    add_217: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_385: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_217, 0.5);  add_217 = None
    mul_386: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_387: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_386, -0.5);  mul_386 = None
    exp_28: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_387);  mul_387 = None
    mul_388: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_389: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, mul_388);  view_197 = mul_388 = None
    add_218: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_385, mul_389);  mul_385 = mul_389 = None
    mul_390: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_510, add_218);  view_510 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_511: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_390, [1568, 1152]);  mul_390 = None
    permute_379: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_101: "f32[1568, 384]" = torch.ops.aten.mm.default(view_511, permute_379);  permute_379 = None
    permute_380: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_102: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_380, view_196);  permute_380 = view_196 = None
    permute_381: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_125: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[1152]" = torch.ops.aten.view.default(sum_125, [1152]);  sum_125 = None
    permute_382: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_513: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_101, [8, 14, 14, 384]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_226: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format);  add_115 = None
    sub_123: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_226, getitem_77);  clone_226 = getitem_77 = None
    mul_391: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_26);  sub_123 = None
    mul_392: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_513, primals_150);  primals_150 = None
    mul_393: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_392, 384)
    sum_126: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [3], True)
    mul_394: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_392, mul_391);  mul_392 = None
    sum_127: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True);  mul_394 = None
    mul_395: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_391, sum_127);  sum_127 = None
    sub_124: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_393, sum_126);  mul_393 = sum_126 = None
    sub_125: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_124, mul_395);  sub_124 = mul_395 = None
    div_37: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 384);  rsqrt_26 = None
    mul_396: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_37, sub_125);  div_37 = sub_125 = None
    mul_397: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_513, mul_391);  mul_391 = None
    sum_128: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1, 2]);  mul_397 = None
    sum_129: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_513, [0, 1, 2]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_219: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_216, mul_396);  add_216 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_514: "f32[1568, 384]" = torch.ops.aten.view.default(add_219, [1568, 384])
    permute_383: "f32[384, 384]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_103: "f32[1568, 384]" = torch.ops.aten.mm.default(view_514, permute_383);  permute_383 = None
    permute_384: "f32[384, 1568]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_104: "f32[384, 384]" = torch.ops.aten.mm.default(permute_384, view_194);  permute_384 = view_194 = None
    permute_385: "f32[384, 384]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_130: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
    view_515: "f32[384]" = torch.ops.aten.view.default(sum_130, [384]);  sum_130 = None
    permute_386: "f32[384, 384]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_516: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_103, [8, 14, 14, 384]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_517: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_516, [8, 196, 12, 32]);  view_516 = None
    permute_387: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_227: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_518: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_227, [96, 196, 32]);  clone_227 = None
    permute_388: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_68: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_388, view_518);  permute_388 = None
    permute_389: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_69: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_518, permute_389);  view_518 = permute_389 = None
    view_519: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 12, 196, 32]);  bmm_68 = None
    view_520: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 12, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_31: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_398: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_520, alias_31);  view_520 = None
    sum_131: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [-1], True)
    mul_399: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_31, sum_131);  alias_31 = sum_131 = None
    sub_126: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_400: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_126, 0.1767766952966369);  sub_126 = None
    view_521: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_400, [96, 196, 196]);  mul_400 = None
    permute_390: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_70: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_390, view_521);  permute_390 = None
    permute_391: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_71: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_521, permute_391);  view_521 = permute_391 = None
    view_522: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_70, [8, 12, 32, 196]);  bmm_70 = None
    view_523: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_71, [8, 12, 196, 32]);  bmm_71 = None
    permute_392: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_522, [0, 1, 3, 2]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_523, permute_392, view_519]);  view_523 = permute_392 = view_519 = None
    view_524: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_11, [3, 8, 12, 196, 32]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_393: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_524, [1, 3, 0, 2, 4]);  view_524 = None
    clone_228: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_525: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_228, [8, 14, 14, 1152]);  clone_228 = None
    view_526: "f32[1568, 1152]" = torch.ops.aten.view.default(view_525, [1568, 1152]);  view_525 = None
    permute_394: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_105: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_394, view_184);  permute_394 = view_184 = None
    permute_395: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    permute_396: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_106: "f32[1568, 384]" = torch.ops.aten.mm.default(view_526, permute_396);  view_526 = permute_396 = None
    view_527: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_106, [8, 14, 14, 384]);  mm_106 = None
    permute_397: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_229: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format);  add_112 = None
    sub_127: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_229, getitem_72);  clone_229 = getitem_72 = None
    mul_401: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_25);  sub_127 = None
    mul_402: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_527, primals_145);  primals_145 = None
    mul_403: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_402, 384)
    sum_132: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [3], True)
    mul_404: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_402, mul_401);  mul_402 = None
    sum_133: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [3], True);  mul_404 = None
    mul_405: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_401, sum_133);  sum_133 = None
    sub_128: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_403, sum_132);  mul_403 = sum_132 = None
    sub_129: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_128, mul_405);  sub_128 = mul_405 = None
    div_38: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 384);  rsqrt_25 = None
    mul_406: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_38, sub_129);  div_38 = sub_129 = None
    mul_407: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_527, mul_401);  mul_401 = None
    sum_134: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1, 2]);  mul_407 = None
    sum_135: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_527, [0, 1, 2]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_220: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_219, mul_406);  add_219 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_528: "f32[1568, 384]" = torch.ops.aten.view.default(add_220, [1568, 384])
    permute_398: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_107: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_528, permute_398);  permute_398 = None
    permute_399: "f32[384, 1568]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_108: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_399, view_182);  permute_399 = view_182 = None
    permute_400: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_136: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[384]" = torch.ops.aten.view.default(sum_136, [384]);  sum_136 = None
    permute_401: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_530: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_107, [8, 14, 14, 1152]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_408: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_29: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_221: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_409: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_221, 0.5);  add_221 = None
    mul_410: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, view_181)
    mul_411: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_410, -0.5);  mul_410 = None
    exp_29: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_411);  mul_411 = None
    mul_412: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_413: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, mul_412);  view_181 = mul_412 = None
    add_222: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_409, mul_413);  mul_409 = mul_413 = None
    mul_414: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_530, add_222);  view_530 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_531: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_414, [1568, 1152]);  mul_414 = None
    permute_402: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_109: "f32[1568, 384]" = torch.ops.aten.mm.default(view_531, permute_402);  permute_402 = None
    permute_403: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_110: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_403, view_180);  permute_403 = view_180 = None
    permute_404: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_137: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[1152]" = torch.ops.aten.view.default(sum_137, [1152]);  sum_137 = None
    permute_405: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_533: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_109, [8, 14, 14, 384]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_230: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format);  add_108 = None
    sub_130: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_230, getitem_70);  clone_230 = getitem_70 = None
    mul_415: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_24);  sub_130 = None
    mul_416: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_533, primals_139);  primals_139 = None
    mul_417: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_416, 384)
    sum_138: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [3], True)
    mul_418: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_416, mul_415);  mul_416 = None
    sum_139: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [3], True);  mul_418 = None
    mul_419: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_415, sum_139);  sum_139 = None
    sub_131: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_417, sum_138);  mul_417 = sum_138 = None
    sub_132: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_131, mul_419);  sub_131 = mul_419 = None
    div_39: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 384);  rsqrt_24 = None
    mul_420: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_39, sub_132);  div_39 = sub_132 = None
    mul_421: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_533, mul_415);  mul_415 = None
    sum_140: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1, 2]);  mul_421 = None
    sum_141: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_533, [0, 1, 2]);  view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_223: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_220, mul_420);  add_220 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_534: "f32[1568, 384]" = torch.ops.aten.view.default(add_223, [1568, 384])
    permute_406: "f32[384, 384]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_111: "f32[1568, 384]" = torch.ops.aten.mm.default(view_534, permute_406);  permute_406 = None
    permute_407: "f32[384, 1568]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_112: "f32[384, 384]" = torch.ops.aten.mm.default(permute_407, view_178);  permute_407 = view_178 = None
    permute_408: "f32[384, 384]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_142: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[384]" = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
    permute_409: "f32[384, 384]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_536: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_111, [8, 14, 14, 384]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_537: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_536, [8, 196, 12, 32]);  view_536 = None
    permute_410: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    clone_231: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_538: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_231, [96, 196, 32]);  clone_231 = None
    permute_411: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    bmm_72: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_411, view_538);  permute_411 = None
    permute_412: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
    bmm_73: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_538, permute_412);  view_538 = permute_412 = None
    view_539: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 12, 196, 32]);  bmm_72 = None
    view_540: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 12, 196, 196]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_32: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_422: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_540, alias_32);  view_540 = None
    sum_143: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [-1], True)
    mul_423: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_32, sum_143);  alias_32 = sum_143 = None
    sub_133: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_424: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_133, 0.1767766952966369);  sub_133 = None
    view_541: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_424, [96, 196, 196]);  mul_424 = None
    permute_413: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_74: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_413, view_541);  permute_413 = None
    permute_414: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_75: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_541, permute_414);  view_541 = permute_414 = None
    view_542: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_74, [8, 12, 32, 196]);  bmm_74 = None
    view_543: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_75, [8, 12, 196, 32]);  bmm_75 = None
    permute_415: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_543, permute_415, view_539]);  view_543 = permute_415 = view_539 = None
    view_544: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_12, [3, 8, 12, 196, 32]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_416: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_544, [1, 3, 0, 2, 4]);  view_544 = None
    clone_232: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_416, memory_format = torch.contiguous_format);  permute_416 = None
    view_545: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_232, [8, 14, 14, 1152]);  clone_232 = None
    view_546: "f32[1568, 1152]" = torch.ops.aten.view.default(view_545, [1568, 1152]);  view_545 = None
    permute_417: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_113: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_417, view_168);  permute_417 = view_168 = None
    permute_418: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    permute_419: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_114: "f32[1568, 384]" = torch.ops.aten.mm.default(view_546, permute_419);  view_546 = permute_419 = None
    view_547: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_114, [8, 14, 14, 384]);  mm_114 = None
    permute_420: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_233: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_105, memory_format = torch.contiguous_format);  add_105 = None
    sub_134: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_233, getitem_65);  clone_233 = getitem_65 = None
    mul_425: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_23);  sub_134 = None
    mul_426: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_547, primals_134);  primals_134 = None
    mul_427: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_426, 384)
    sum_144: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [3], True)
    mul_428: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_426, mul_425);  mul_426 = None
    sum_145: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [3], True);  mul_428 = None
    mul_429: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_425, sum_145);  sum_145 = None
    sub_135: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_427, sum_144);  mul_427 = sum_144 = None
    sub_136: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_135, mul_429);  sub_135 = mul_429 = None
    div_40: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 384);  rsqrt_23 = None
    mul_430: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_40, sub_136);  div_40 = sub_136 = None
    mul_431: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_547, mul_425);  mul_425 = None
    sum_146: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1, 2]);  mul_431 = None
    sum_147: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_547, [0, 1, 2]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_224: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_223, mul_430);  add_223 = mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_548: "f32[1568, 384]" = torch.ops.aten.view.default(add_224, [1568, 384])
    permute_421: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_115: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_548, permute_421);  permute_421 = None
    permute_422: "f32[384, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_116: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_422, view_166);  permute_422 = view_166 = None
    permute_423: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_148: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[384]" = torch.ops.aten.view.default(sum_148, [384]);  sum_148 = None
    permute_424: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_550: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_115, [8, 14, 14, 1152]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_432: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.7071067811865476)
    erf_30: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_432);  mul_432 = None
    add_225: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_433: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
    mul_434: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, view_165)
    mul_435: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_434, -0.5);  mul_434 = None
    exp_30: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_435);  mul_435 = None
    mul_436: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_437: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, mul_436);  view_165 = mul_436 = None
    add_226: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_433, mul_437);  mul_433 = mul_437 = None
    mul_438: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_550, add_226);  view_550 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_551: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_438, [1568, 1152]);  mul_438 = None
    permute_425: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_117: "f32[1568, 384]" = torch.ops.aten.mm.default(view_551, permute_425);  permute_425 = None
    permute_426: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_118: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_426, view_164);  permute_426 = view_164 = None
    permute_427: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_149: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[1152]" = torch.ops.aten.view.default(sum_149, [1152]);  sum_149 = None
    permute_428: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_553: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_117, [8, 14, 14, 384]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_234: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format);  add_101 = None
    sub_137: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_234, getitem_63);  clone_234 = getitem_63 = None
    mul_439: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_22);  sub_137 = None
    mul_440: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_553, primals_128);  primals_128 = None
    mul_441: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_440, 384)
    sum_150: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [3], True)
    mul_442: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_440, mul_439);  mul_440 = None
    sum_151: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [3], True);  mul_442 = None
    mul_443: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_439, sum_151);  sum_151 = None
    sub_138: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_441, sum_150);  mul_441 = sum_150 = None
    sub_139: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_138, mul_443);  sub_138 = mul_443 = None
    div_41: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 384);  rsqrt_22 = None
    mul_444: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_41, sub_139);  div_41 = sub_139 = None
    mul_445: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_553, mul_439);  mul_439 = None
    sum_152: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1, 2]);  mul_445 = None
    sum_153: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_553, [0, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_227: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_224, mul_444);  add_224 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_554: "f32[1568, 384]" = torch.ops.aten.view.default(add_227, [1568, 384])
    permute_429: "f32[384, 384]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_119: "f32[1568, 384]" = torch.ops.aten.mm.default(view_554, permute_429);  permute_429 = None
    permute_430: "f32[384, 1568]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_120: "f32[384, 384]" = torch.ops.aten.mm.default(permute_430, view_162);  permute_430 = view_162 = None
    permute_431: "f32[384, 384]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_154: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[384]" = torch.ops.aten.view.default(sum_154, [384]);  sum_154 = None
    permute_432: "f32[384, 384]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_556: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_119, [8, 14, 14, 384]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_557: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_556, [8, 196, 12, 32]);  view_556 = None
    permute_433: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
    clone_235: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_558: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_235, [96, 196, 32]);  clone_235 = None
    permute_434: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    bmm_76: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_434, view_558);  permute_434 = None
    permute_435: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
    bmm_77: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_558, permute_435);  view_558 = permute_435 = None
    view_559: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 12, 196, 32]);  bmm_76 = None
    view_560: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 12, 196, 196]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_33: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_446: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_560, alias_33);  view_560 = None
    sum_155: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [-1], True)
    mul_447: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_33, sum_155);  alias_33 = sum_155 = None
    sub_140: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_448: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_140, 0.1767766952966369);  sub_140 = None
    view_561: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_448, [96, 196, 196]);  mul_448 = None
    permute_436: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    bmm_78: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_436, view_561);  permute_436 = None
    permute_437: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    bmm_79: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_561, permute_437);  view_561 = permute_437 = None
    view_562: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_78, [8, 12, 32, 196]);  bmm_78 = None
    view_563: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_79, [8, 12, 196, 32]);  bmm_79 = None
    permute_438: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_562, [0, 1, 3, 2]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_13: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_563, permute_438, view_559]);  view_563 = permute_438 = view_559 = None
    view_564: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_13, [3, 8, 12, 196, 32]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_439: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_564, [1, 3, 0, 2, 4]);  view_564 = None
    clone_236: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_565: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_236, [8, 14, 14, 1152]);  clone_236 = None
    view_566: "f32[1568, 1152]" = torch.ops.aten.view.default(view_565, [1568, 1152]);  view_565 = None
    permute_440: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_121: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_440, view_152);  permute_440 = view_152 = None
    permute_441: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    permute_442: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_122: "f32[1568, 384]" = torch.ops.aten.mm.default(view_566, permute_442);  view_566 = permute_442 = None
    view_567: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_122, [8, 14, 14, 384]);  mm_122 = None
    permute_443: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_237: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format);  add_98 = None
    sub_141: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_237, getitem_58);  clone_237 = getitem_58 = None
    mul_449: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_21);  sub_141 = None
    mul_450: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_567, primals_123);  primals_123 = None
    mul_451: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_450, 384)
    sum_156: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [3], True)
    mul_452: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_450, mul_449);  mul_450 = None
    sum_157: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [3], True);  mul_452 = None
    mul_453: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_449, sum_157);  sum_157 = None
    sub_142: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_451, sum_156);  mul_451 = sum_156 = None
    sub_143: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_142, mul_453);  sub_142 = mul_453 = None
    div_42: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None
    mul_454: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_42, sub_143);  div_42 = sub_143 = None
    mul_455: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_567, mul_449);  mul_449 = None
    sum_158: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1, 2]);  mul_455 = None
    sum_159: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_567, [0, 1, 2]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_228: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_227, mul_454);  add_227 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_568: "f32[1568, 384]" = torch.ops.aten.view.default(add_228, [1568, 384])
    permute_444: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_123: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_568, permute_444);  permute_444 = None
    permute_445: "f32[384, 1568]" = torch.ops.aten.permute.default(view_568, [1, 0])
    mm_124: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_445, view_150);  permute_445 = view_150 = None
    permute_446: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_160: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
    view_569: "f32[384]" = torch.ops.aten.view.default(sum_160, [384]);  sum_160 = None
    permute_447: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_570: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_123, [8, 14, 14, 1152]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_456: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476)
    erf_31: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_456);  mul_456 = None
    add_229: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_457: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_229, 0.5);  add_229 = None
    mul_458: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, view_149)
    mul_459: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
    exp_31: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
    mul_460: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_461: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, mul_460);  view_149 = mul_460 = None
    add_230: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
    mul_462: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_570, add_230);  view_570 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_571: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_462, [1568, 1152]);  mul_462 = None
    permute_448: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_125: "f32[1568, 384]" = torch.ops.aten.mm.default(view_571, permute_448);  permute_448 = None
    permute_449: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_126: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_449, view_148);  permute_449 = view_148 = None
    permute_450: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_161: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[1152]" = torch.ops.aten.view.default(sum_161, [1152]);  sum_161 = None
    permute_451: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_573: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_125, [8, 14, 14, 384]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_238: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format);  add_94 = None
    sub_144: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_238, getitem_56);  clone_238 = getitem_56 = None
    mul_463: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_20);  sub_144 = None
    mul_464: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_573, primals_117);  primals_117 = None
    mul_465: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_464, 384)
    sum_162: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [3], True)
    mul_466: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_464, mul_463);  mul_464 = None
    sum_163: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [3], True);  mul_466 = None
    mul_467: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_463, sum_163);  sum_163 = None
    sub_145: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_465, sum_162);  mul_465 = sum_162 = None
    sub_146: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_145, mul_467);  sub_145 = mul_467 = None
    div_43: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None
    mul_468: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_43, sub_146);  div_43 = sub_146 = None
    mul_469: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_573, mul_463);  mul_463 = None
    sum_164: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1, 2]);  mul_469 = None
    sum_165: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_573, [0, 1, 2]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_231: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_228, mul_468);  add_228 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_574: "f32[1568, 384]" = torch.ops.aten.view.default(add_231, [1568, 384])
    permute_452: "f32[384, 384]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_127: "f32[1568, 384]" = torch.ops.aten.mm.default(view_574, permute_452);  permute_452 = None
    permute_453: "f32[384, 1568]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_128: "f32[384, 384]" = torch.ops.aten.mm.default(permute_453, view_146);  permute_453 = view_146 = None
    permute_454: "f32[384, 384]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_166: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[384]" = torch.ops.aten.view.default(sum_166, [384]);  sum_166 = None
    permute_455: "f32[384, 384]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_576: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_127, [8, 14, 14, 384]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_577: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_576, [8, 196, 12, 32]);  view_576 = None
    permute_456: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
    clone_239: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
    view_578: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_239, [96, 196, 32]);  clone_239 = None
    permute_457: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_80: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_457, view_578);  permute_457 = None
    permute_458: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_81: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_578, permute_458);  view_578 = permute_458 = None
    view_579: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 12, 196, 32]);  bmm_80 = None
    view_580: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 12, 196, 196]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_34: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_470: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_580, alias_34);  view_580 = None
    sum_167: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [-1], True)
    mul_471: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_34, sum_167);  alias_34 = sum_167 = None
    sub_147: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_472: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_147, 0.1767766952966369);  sub_147 = None
    view_581: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_472, [96, 196, 196]);  mul_472 = None
    permute_459: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    bmm_82: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_459, view_581);  permute_459 = None
    permute_460: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    bmm_83: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_581, permute_460);  view_581 = permute_460 = None
    view_582: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_82, [8, 12, 32, 196]);  bmm_82 = None
    view_583: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_83, [8, 12, 196, 32]);  bmm_83 = None
    permute_461: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_582, [0, 1, 3, 2]);  view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_14: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_583, permute_461, view_579]);  view_583 = permute_461 = view_579 = None
    view_584: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_14, [3, 8, 12, 196, 32]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_462: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_584, [1, 3, 0, 2, 4]);  view_584 = None
    clone_240: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
    view_585: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_240, [8, 14, 14, 1152]);  clone_240 = None
    view_586: "f32[1568, 1152]" = torch.ops.aten.view.default(view_585, [1568, 1152]);  view_585 = None
    permute_463: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_129: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_463, view_136);  permute_463 = view_136 = None
    permute_464: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    permute_465: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_130: "f32[1568, 384]" = torch.ops.aten.mm.default(view_586, permute_465);  view_586 = permute_465 = None
    view_587: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_130, [8, 14, 14, 384]);  mm_130 = None
    permute_466: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_241: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_91, memory_format = torch.contiguous_format);  add_91 = None
    sub_148: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_241, getitem_51);  clone_241 = getitem_51 = None
    mul_473: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_19);  sub_148 = None
    mul_474: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_587, primals_112);  primals_112 = None
    mul_475: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_474, 384)
    sum_168: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [3], True)
    mul_476: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_474, mul_473);  mul_474 = None
    sum_169: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [3], True);  mul_476 = None
    mul_477: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_473, sum_169);  sum_169 = None
    sub_149: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_475, sum_168);  mul_475 = sum_168 = None
    sub_150: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_149, mul_477);  sub_149 = mul_477 = None
    div_44: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 384);  rsqrt_19 = None
    mul_478: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_44, sub_150);  div_44 = sub_150 = None
    mul_479: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_587, mul_473);  mul_473 = None
    sum_170: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1, 2]);  mul_479 = None
    sum_171: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_587, [0, 1, 2]);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_232: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_231, mul_478);  add_231 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_588: "f32[1568, 384]" = torch.ops.aten.view.default(add_232, [1568, 384])
    permute_467: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_131: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_588, permute_467);  permute_467 = None
    permute_468: "f32[384, 1568]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_132: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_468, view_134);  permute_468 = view_134 = None
    permute_469: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_172: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[384]" = torch.ops.aten.view.default(sum_172, [384]);  sum_172 = None
    permute_470: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_590: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_131, [8, 14, 14, 1152]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_480: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
    erf_32: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_480);  mul_480 = None
    add_233: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_481: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_233, 0.5);  add_233 = None
    mul_482: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, view_133)
    mul_483: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_482, -0.5);  mul_482 = None
    exp_32: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_483);  mul_483 = None
    mul_484: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_485: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, mul_484);  view_133 = mul_484 = None
    add_234: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_481, mul_485);  mul_481 = mul_485 = None
    mul_486: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_590, add_234);  view_590 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_591: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_486, [1568, 1152]);  mul_486 = None
    permute_471: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_133: "f32[1568, 384]" = torch.ops.aten.mm.default(view_591, permute_471);  permute_471 = None
    permute_472: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_134: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_472, view_132);  permute_472 = view_132 = None
    permute_473: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_173: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[1152]" = torch.ops.aten.view.default(sum_173, [1152]);  sum_173 = None
    permute_474: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_593: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_133, [8, 14, 14, 384]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_242: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format);  add_87 = None
    sub_151: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_242, getitem_49);  clone_242 = getitem_49 = None
    mul_487: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_18);  sub_151 = None
    mul_488: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_593, primals_106);  primals_106 = None
    mul_489: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_488, 384)
    sum_174: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [3], True)
    mul_490: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_488, mul_487);  mul_488 = None
    sum_175: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [3], True);  mul_490 = None
    mul_491: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_487, sum_175);  sum_175 = None
    sub_152: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_489, sum_174);  mul_489 = sum_174 = None
    sub_153: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_152, mul_491);  sub_152 = mul_491 = None
    div_45: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 384);  rsqrt_18 = None
    mul_492: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_45, sub_153);  div_45 = sub_153 = None
    mul_493: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_593, mul_487);  mul_487 = None
    sum_176: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1, 2]);  mul_493 = None
    sum_177: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_593, [0, 1, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_235: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_232, mul_492);  add_232 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_594: "f32[1568, 384]" = torch.ops.aten.view.default(add_235, [1568, 384])
    permute_475: "f32[384, 384]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_135: "f32[1568, 384]" = torch.ops.aten.mm.default(view_594, permute_475);  permute_475 = None
    permute_476: "f32[384, 1568]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_136: "f32[384, 384]" = torch.ops.aten.mm.default(permute_476, view_130);  permute_476 = view_130 = None
    permute_477: "f32[384, 384]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_178: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[384]" = torch.ops.aten.view.default(sum_178, [384]);  sum_178 = None
    permute_478: "f32[384, 384]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_596: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_135, [8, 14, 14, 384]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_597: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_596, [8, 196, 12, 32]);  view_596 = None
    permute_479: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_243: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_598: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_243, [96, 196, 32]);  clone_243 = None
    permute_480: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    bmm_84: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_480, view_598);  permute_480 = None
    permute_481: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
    bmm_85: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_598, permute_481);  view_598 = permute_481 = None
    view_599: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_84, [8, 12, 196, 32]);  bmm_84 = None
    view_600: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_85, [8, 12, 196, 196]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_35: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_494: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_600, alias_35);  view_600 = None
    sum_179: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [-1], True)
    mul_495: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_35, sum_179);  alias_35 = sum_179 = None
    sub_154: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_496: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_154, 0.1767766952966369);  sub_154 = None
    view_601: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_496, [96, 196, 196]);  mul_496 = None
    permute_482: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_86: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_482, view_601);  permute_482 = None
    permute_483: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_87: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_601, permute_483);  view_601 = permute_483 = None
    view_602: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_86, [8, 12, 32, 196]);  bmm_86 = None
    view_603: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_87, [8, 12, 196, 32]);  bmm_87 = None
    permute_484: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 2]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_603, permute_484, view_599]);  view_603 = permute_484 = view_599 = None
    view_604: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_15, [3, 8, 12, 196, 32]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_485: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_604, [1, 3, 0, 2, 4]);  view_604 = None
    clone_244: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_605: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_244, [8, 14, 14, 1152]);  clone_244 = None
    view_606: "f32[1568, 1152]" = torch.ops.aten.view.default(view_605, [1568, 1152]);  view_605 = None
    permute_486: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_137: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_486, view_120);  permute_486 = view_120 = None
    permute_487: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    permute_488: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_138: "f32[1568, 384]" = torch.ops.aten.mm.default(view_606, permute_488);  view_606 = permute_488 = None
    view_607: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_138, [8, 14, 14, 384]);  mm_138 = None
    permute_489: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_245: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_84, memory_format = torch.contiguous_format);  add_84 = None
    sub_155: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_245, getitem_44);  clone_245 = getitem_44 = None
    mul_497: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_17);  sub_155 = None
    mul_498: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_607, primals_101);  primals_101 = None
    mul_499: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_498, 384)
    sum_180: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [3], True)
    mul_500: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_498, mul_497);  mul_498 = None
    sum_181: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [3], True);  mul_500 = None
    mul_501: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_497, sum_181);  sum_181 = None
    sub_156: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_499, sum_180);  mul_499 = sum_180 = None
    sub_157: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_156, mul_501);  sub_156 = mul_501 = None
    div_46: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 384);  rsqrt_17 = None
    mul_502: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_46, sub_157);  div_46 = sub_157 = None
    mul_503: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_607, mul_497);  mul_497 = None
    sum_182: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1, 2]);  mul_503 = None
    sum_183: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_607, [0, 1, 2]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_236: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_235, mul_502);  add_235 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_608: "f32[1568, 384]" = torch.ops.aten.view.default(add_236, [1568, 384])
    permute_490: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_139: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_608, permute_490);  permute_490 = None
    permute_491: "f32[384, 1568]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_140: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_491, view_118);  permute_491 = view_118 = None
    permute_492: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
    view_609: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_493: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_610: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_139, [8, 14, 14, 1152]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_504: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_33: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_504);  mul_504 = None
    add_237: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_505: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_237, 0.5);  add_237 = None
    mul_506: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, view_117)
    mul_507: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_506, -0.5);  mul_506 = None
    exp_33: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_507);  mul_507 = None
    mul_508: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_509: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, mul_508);  view_117 = mul_508 = None
    add_238: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_505, mul_509);  mul_505 = mul_509 = None
    mul_510: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_610, add_238);  view_610 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_611: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_510, [1568, 1152]);  mul_510 = None
    permute_494: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_141: "f32[1568, 384]" = torch.ops.aten.mm.default(view_611, permute_494);  permute_494 = None
    permute_495: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_142: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_495, view_116);  permute_495 = view_116 = None
    permute_496: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_185: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[1152]" = torch.ops.aten.view.default(sum_185, [1152]);  sum_185 = None
    permute_497: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    view_613: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_141, [8, 14, 14, 384]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_246: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format);  add_80 = None
    sub_158: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_246, getitem_42);  clone_246 = getitem_42 = None
    mul_511: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_16);  sub_158 = None
    mul_512: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_613, primals_95);  primals_95 = None
    mul_513: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_512, 384)
    sum_186: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [3], True)
    mul_514: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_512, mul_511);  mul_512 = None
    sum_187: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [3], True);  mul_514 = None
    mul_515: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_511, sum_187);  sum_187 = None
    sub_159: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_513, sum_186);  mul_513 = sum_186 = None
    sub_160: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_159, mul_515);  sub_159 = mul_515 = None
    div_47: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None
    mul_516: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_47, sub_160);  div_47 = sub_160 = None
    mul_517: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_613, mul_511);  mul_511 = None
    sum_188: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1, 2]);  mul_517 = None
    sum_189: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_613, [0, 1, 2]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_239: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_236, mul_516);  add_236 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_614: "f32[1568, 384]" = torch.ops.aten.view.default(add_239, [1568, 384])
    permute_498: "f32[384, 384]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_143: "f32[1568, 384]" = torch.ops.aten.mm.default(view_614, permute_498);  permute_498 = None
    permute_499: "f32[384, 1568]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_144: "f32[384, 384]" = torch.ops.aten.mm.default(permute_499, view_114);  permute_499 = view_114 = None
    permute_500: "f32[384, 384]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_190: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[384]" = torch.ops.aten.view.default(sum_190, [384]);  sum_190 = None
    permute_501: "f32[384, 384]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_616: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_143, [8, 14, 14, 384]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_617: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_616, [8, 196, 12, 32]);  view_616 = None
    permute_502: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
    clone_247: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
    view_618: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_247, [96, 196, 32]);  clone_247 = None
    permute_503: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    bmm_88: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_503, view_618);  permute_503 = None
    permute_504: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_89: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_618, permute_504);  view_618 = permute_504 = None
    view_619: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_88, [8, 12, 196, 32]);  bmm_88 = None
    view_620: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_89, [8, 12, 196, 196]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_36: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_518: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_620, alias_36);  view_620 = None
    sum_191: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [-1], True)
    mul_519: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_36, sum_191);  alias_36 = sum_191 = None
    sub_161: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_520: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_161, 0.1767766952966369);  sub_161 = None
    view_621: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_520, [96, 196, 196]);  mul_520 = None
    permute_505: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    bmm_90: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_505, view_621);  permute_505 = None
    permute_506: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
    bmm_91: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_621, permute_506);  view_621 = permute_506 = None
    view_622: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_90, [8, 12, 32, 196]);  bmm_90 = None
    view_623: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_91, [8, 12, 196, 32]);  bmm_91 = None
    permute_507: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_622, [0, 1, 3, 2]);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_16: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_623, permute_507, view_619]);  view_623 = permute_507 = view_619 = None
    view_624: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_16, [3, 8, 12, 196, 32]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_508: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_624, [1, 3, 0, 2, 4]);  view_624 = None
    clone_248: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_508, memory_format = torch.contiguous_format);  permute_508 = None
    view_625: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_248, [8, 14, 14, 1152]);  clone_248 = None
    view_626: "f32[1568, 1152]" = torch.ops.aten.view.default(view_625, [1568, 1152]);  view_625 = None
    permute_509: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_626, [1, 0])
    mm_145: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_509, view_104);  permute_509 = view_104 = None
    permute_510: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    permute_511: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_146: "f32[1568, 384]" = torch.ops.aten.mm.default(view_626, permute_511);  view_626 = permute_511 = None
    view_627: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_146, [8, 14, 14, 384]);  mm_146 = None
    permute_512: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_249: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format);  add_77 = None
    sub_162: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_249, getitem_37);  clone_249 = getitem_37 = None
    mul_521: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_15);  sub_162 = None
    mul_522: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_627, primals_90);  primals_90 = None
    mul_523: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_522, 384)
    sum_192: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [3], True)
    mul_524: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_522, mul_521);  mul_522 = None
    sum_193: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [3], True);  mul_524 = None
    mul_525: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_521, sum_193);  sum_193 = None
    sub_163: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_523, sum_192);  mul_523 = sum_192 = None
    sub_164: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_163, mul_525);  sub_163 = mul_525 = None
    div_48: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None
    mul_526: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_48, sub_164);  div_48 = sub_164 = None
    mul_527: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_627, mul_521);  mul_521 = None
    sum_194: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1, 2]);  mul_527 = None
    sum_195: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_627, [0, 1, 2]);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_240: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_239, mul_526);  add_239 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_628: "f32[1568, 384]" = torch.ops.aten.view.default(add_240, [1568, 384])
    permute_513: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_147: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_628, permute_513);  permute_513 = None
    permute_514: "f32[384, 1568]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_148: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_514, view_102);  permute_514 = view_102 = None
    permute_515: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_196: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[384]" = torch.ops.aten.view.default(sum_196, [384]);  sum_196 = None
    permute_516: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_630: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_147, [8, 14, 14, 1152]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_528: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_34: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_528);  mul_528 = None
    add_241: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_529: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_241, 0.5);  add_241 = None
    mul_530: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, view_101)
    mul_531: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_530, -0.5);  mul_530 = None
    exp_34: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_531);  mul_531 = None
    mul_532: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_533: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, mul_532);  view_101 = mul_532 = None
    add_242: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_529, mul_533);  mul_529 = mul_533 = None
    mul_534: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_630, add_242);  view_630 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_631: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_534, [1568, 1152]);  mul_534 = None
    permute_517: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_149: "f32[1568, 384]" = torch.ops.aten.mm.default(view_631, permute_517);  permute_517 = None
    permute_518: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_150: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_518, view_100);  permute_518 = view_100 = None
    permute_519: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_197: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[1152]" = torch.ops.aten.view.default(sum_197, [1152]);  sum_197 = None
    permute_520: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_633: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_149, [8, 14, 14, 384]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_250: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format);  add_73 = None
    sub_165: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_250, getitem_35);  clone_250 = getitem_35 = None
    mul_535: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_14);  sub_165 = None
    mul_536: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_633, primals_84);  primals_84 = None
    mul_537: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_536, 384)
    sum_198: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [3], True)
    mul_538: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_536, mul_535);  mul_536 = None
    sum_199: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [3], True);  mul_538 = None
    mul_539: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_535, sum_199);  sum_199 = None
    sub_166: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_537, sum_198);  mul_537 = sum_198 = None
    sub_167: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_166, mul_539);  sub_166 = mul_539 = None
    div_49: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 384);  rsqrt_14 = None
    mul_540: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_49, sub_167);  div_49 = sub_167 = None
    mul_541: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_633, mul_535);  mul_535 = None
    sum_200: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1, 2]);  mul_541 = None
    sum_201: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_633, [0, 1, 2]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_243: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_240, mul_540);  add_240 = mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_634: "f32[1568, 384]" = torch.ops.aten.view.default(add_243, [1568, 384])
    permute_521: "f32[384, 384]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_151: "f32[1568, 384]" = torch.ops.aten.mm.default(view_634, permute_521);  permute_521 = None
    permute_522: "f32[384, 1568]" = torch.ops.aten.permute.default(view_634, [1, 0])
    mm_152: "f32[384, 384]" = torch.ops.aten.mm.default(permute_522, view_98);  permute_522 = view_98 = None
    permute_523: "f32[384, 384]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_202: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True);  view_634 = None
    view_635: "f32[384]" = torch.ops.aten.view.default(sum_202, [384]);  sum_202 = None
    permute_524: "f32[384, 384]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_636: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_151, [8, 14, 14, 384]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_637: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_636, [8, 196, 12, 32]);  view_636 = None
    permute_525: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
    clone_251: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_638: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_251, [96, 196, 32]);  clone_251 = None
    permute_526: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    bmm_92: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_526, view_638);  permute_526 = None
    permute_527: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    bmm_93: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_638, permute_527);  view_638 = permute_527 = None
    view_639: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_92, [8, 12, 196, 32]);  bmm_92 = None
    view_640: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_93, [8, 12, 196, 196]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_37: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_542: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_640, alias_37);  view_640 = None
    sum_203: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [-1], True)
    mul_543: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_37, sum_203);  alias_37 = sum_203 = None
    sub_168: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_544: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_168, 0.1767766952966369);  sub_168 = None
    view_641: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_544, [96, 196, 196]);  mul_544 = None
    permute_528: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_94: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_528, view_641);  permute_528 = None
    permute_529: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    bmm_95: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_641, permute_529);  view_641 = permute_529 = None
    view_642: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_94, [8, 12, 32, 196]);  bmm_94 = None
    view_643: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_95, [8, 12, 196, 32]);  bmm_95 = None
    permute_530: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_642, [0, 1, 3, 2]);  view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_17: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_643, permute_530, view_639]);  view_643 = permute_530 = view_639 = None
    view_644: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_17, [3, 8, 12, 196, 32]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_531: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_644, [1, 3, 0, 2, 4]);  view_644 = None
    clone_252: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
    view_645: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_252, [8, 14, 14, 1152]);  clone_252 = None
    view_646: "f32[1568, 1152]" = torch.ops.aten.view.default(view_645, [1568, 1152]);  view_645 = None
    permute_532: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_153: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_532, view_88);  permute_532 = view_88 = None
    permute_533: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    permute_534: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_154: "f32[1568, 384]" = torch.ops.aten.mm.default(view_646, permute_534);  view_646 = permute_534 = None
    view_647: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_154, [8, 14, 14, 384]);  mm_154 = None
    permute_535: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_253: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format);  add_70 = None
    sub_169: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_253, getitem_30);  clone_253 = getitem_30 = None
    mul_545: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_13);  sub_169 = None
    mul_546: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_647, primals_79);  primals_79 = None
    mul_547: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_546, 384)
    sum_204: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [3], True)
    mul_548: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_546, mul_545);  mul_546 = None
    sum_205: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [3], True);  mul_548 = None
    mul_549: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_545, sum_205);  sum_205 = None
    sub_170: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_547, sum_204);  mul_547 = sum_204 = None
    sub_171: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_170, mul_549);  sub_170 = mul_549 = None
    div_50: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 384);  rsqrt_13 = None
    mul_550: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_50, sub_171);  div_50 = sub_171 = None
    mul_551: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_647, mul_545);  mul_545 = None
    sum_206: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 1, 2]);  mul_551 = None
    sum_207: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_647, [0, 1, 2]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_244: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_243, mul_550);  add_243 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_648: "f32[1568, 384]" = torch.ops.aten.view.default(add_244, [1568, 384])
    permute_536: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_155: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_648, permute_536);  permute_536 = None
    permute_537: "f32[384, 1568]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_156: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_537, view_86);  permute_537 = view_86 = None
    permute_538: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_208: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[384]" = torch.ops.aten.view.default(sum_208, [384]);  sum_208 = None
    permute_539: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_650: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_155, [8, 14, 14, 1152]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_552: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_35: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_552);  mul_552 = None
    add_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_553: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_554: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_555: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_554, -0.5);  mul_554 = None
    exp_35: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_555);  mul_555 = None
    mul_556: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_557: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, mul_556);  view_85 = mul_556 = None
    add_246: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_553, mul_557);  mul_553 = mul_557 = None
    mul_558: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_650, add_246);  view_650 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_651: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_558, [1568, 1152]);  mul_558 = None
    permute_540: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_157: "f32[1568, 384]" = torch.ops.aten.mm.default(view_651, permute_540);  permute_540 = None
    permute_541: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_651, [1, 0])
    mm_158: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_541, view_84);  permute_541 = view_84 = None
    permute_542: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_209: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_651, [0], True);  view_651 = None
    view_652: "f32[1152]" = torch.ops.aten.view.default(sum_209, [1152]);  sum_209 = None
    permute_543: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_653: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_157, [8, 14, 14, 384]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_254: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format);  add_66 = None
    sub_172: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_254, getitem_28);  clone_254 = getitem_28 = None
    mul_559: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_12);  sub_172 = None
    mul_560: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_653, primals_73);  primals_73 = None
    mul_561: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_560, 384)
    sum_210: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [3], True)
    mul_562: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_560, mul_559);  mul_560 = None
    sum_211: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [3], True);  mul_562 = None
    mul_563: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_559, sum_211);  sum_211 = None
    sub_173: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_561, sum_210);  mul_561 = sum_210 = None
    sub_174: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_173, mul_563);  sub_173 = mul_563 = None
    div_51: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 384);  rsqrt_12 = None
    mul_564: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_51, sub_174);  div_51 = sub_174 = None
    mul_565: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_653, mul_559);  mul_559 = None
    sum_212: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1, 2]);  mul_565 = None
    sum_213: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1, 2]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_247: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_244, mul_564);  add_244 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_654: "f32[1568, 384]" = torch.ops.aten.view.default(add_247, [1568, 384])
    permute_544: "f32[384, 384]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_159: "f32[1568, 384]" = torch.ops.aten.mm.default(view_654, permute_544);  permute_544 = None
    permute_545: "f32[384, 1568]" = torch.ops.aten.permute.default(view_654, [1, 0])
    mm_160: "f32[384, 384]" = torch.ops.aten.mm.default(permute_545, view_82);  permute_545 = view_82 = None
    permute_546: "f32[384, 384]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_214: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[384]" = torch.ops.aten.view.default(sum_214, [384]);  sum_214 = None
    permute_547: "f32[384, 384]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_656: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_159, [8, 14, 14, 384]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_657: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_656, [8, 196, 12, 32]);  view_656 = None
    permute_548: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    clone_255: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
    view_658: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_255, [96, 196, 32]);  clone_255 = None
    permute_549: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_96: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_549, view_658);  permute_549 = None
    permute_550: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_97: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_658, permute_550);  view_658 = permute_550 = None
    view_659: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_96, [8, 12, 196, 32]);  bmm_96 = None
    view_660: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_97, [8, 12, 196, 196]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_38: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_566: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_660, alias_38);  view_660 = None
    sum_215: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [-1], True)
    mul_567: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_38, sum_215);  alias_38 = sum_215 = None
    sub_175: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_568: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_175, 0.1767766952966369);  sub_175 = None
    view_661: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_568, [96, 196, 196]);  mul_568 = None
    permute_551: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_98: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_551, view_661);  permute_551 = None
    permute_552: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_99: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_661, permute_552);  view_661 = permute_552 = None
    view_662: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_98, [8, 12, 32, 196]);  bmm_98 = None
    view_663: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_99, [8, 12, 196, 32]);  bmm_99 = None
    permute_553: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_662, [0, 1, 3, 2]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_18: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_663, permute_553, view_659]);  view_663 = permute_553 = view_659 = None
    view_664: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 12, 196, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_554: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_664, [1, 3, 0, 2, 4]);  view_664 = None
    clone_256: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
    view_665: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_256, [8, 14, 14, 1152]);  clone_256 = None
    view_666: "f32[1568, 1152]" = torch.ops.aten.view.default(view_665, [1568, 1152]);  view_665 = None
    permute_555: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_161: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_555, view_72);  permute_555 = view_72 = None
    permute_556: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    permute_557: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_162: "f32[1568, 384]" = torch.ops.aten.mm.default(view_666, permute_557);  view_666 = permute_557 = None
    view_667: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_162, [8, 14, 14, 384]);  mm_162 = None
    permute_558: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_257: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(clone_44, memory_format = torch.contiguous_format);  clone_44 = None
    sub_176: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_257, getitem_23);  clone_257 = getitem_23 = None
    mul_569: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_176, rsqrt_11);  sub_176 = None
    mul_570: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_667, primals_68);  primals_68 = None
    mul_571: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_570, 384)
    sum_216: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [3], True)
    mul_572: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_570, mul_569);  mul_570 = None
    sum_217: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [3], True);  mul_572 = None
    mul_573: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_569, sum_217);  sum_217 = None
    sub_177: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_571, sum_216);  mul_571 = sum_216 = None
    sub_178: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_177, mul_573);  sub_177 = mul_573 = None
    div_52: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 384);  rsqrt_11 = None
    mul_574: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_52, sub_178);  div_52 = sub_178 = None
    mul_575: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_667, mul_569);  mul_569 = None
    sum_218: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1, 2]);  mul_575 = None
    sum_219: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_667, [0, 1, 2]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_248: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_247, mul_574);  add_247 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    sum_220: "f32[1, 14, 14, 384]" = torch.ops.aten.sum.dim_IntList(add_248, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    permute_559: "f32[8, 384, 14, 14]" = torch.ops.aten.permute.default(add_248, [0, 3, 1, 2]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    sum_221: "f32[384]" = torch.ops.aten.sum.dim_IntList(permute_559, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_559, permute_57, primals_66, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_559 = permute_57 = primals_66 = None
    getitem_136: "f32[8, 192, 28, 28]" = convolution_backward[0]
    getitem_137: "f32[384, 192, 2, 2]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    permute_560: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 3, 1]);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_258: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format)
    view_668: "f32[6272, 192]" = torch.ops.aten.view.default(clone_258, [6272, 192]);  clone_258 = None
    permute_561: "f32[192, 576]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_163: "f32[6272, 576]" = torch.ops.aten.mm.default(view_668, permute_561);  permute_561 = None
    permute_562: "f32[192, 6272]" = torch.ops.aten.permute.default(view_668, [1, 0])
    mm_164: "f32[192, 576]" = torch.ops.aten.mm.default(permute_562, view_70);  permute_562 = view_70 = None
    permute_563: "f32[576, 192]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_222: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
    view_669: "f32[192]" = torch.ops.aten.view.default(sum_222, [192]);  sum_222 = None
    permute_564: "f32[192, 576]" = torch.ops.aten.permute.default(permute_563, [1, 0]);  permute_563 = None
    view_670: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_163, [8, 28, 28, 576]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_576: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_36: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_576);  mul_576 = None
    add_249: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_577: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_249, 0.5);  add_249 = None
    mul_578: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_579: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_578, -0.5);  mul_578 = None
    exp_36: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_579);  mul_579 = None
    mul_580: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_581: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, mul_580);  view_69 = mul_580 = None
    add_250: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_577, mul_581);  mul_577 = mul_581 = None
    mul_582: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_670, add_250);  view_670 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_671: "f32[6272, 576]" = torch.ops.aten.view.default(mul_582, [6272, 576]);  mul_582 = None
    permute_565: "f32[576, 192]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_165: "f32[6272, 192]" = torch.ops.aten.mm.default(view_671, permute_565);  permute_565 = None
    permute_566: "f32[576, 6272]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_166: "f32[576, 192]" = torch.ops.aten.mm.default(permute_566, view_68);  permute_566 = view_68 = None
    permute_567: "f32[192, 576]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_223: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[576]" = torch.ops.aten.view.default(sum_223, [576]);  sum_223 = None
    permute_568: "f32[576, 192]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    view_673: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_165, [8, 28, 28, 192]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_259: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format);  add_58 = None
    sub_179: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_259, getitem_21);  clone_259 = getitem_21 = None
    mul_583: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_179, rsqrt_10);  sub_179 = None
    mul_584: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_673, primals_60);  primals_60 = None
    mul_585: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_584, 192)
    sum_224: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [3], True)
    mul_586: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_584, mul_583);  mul_584 = None
    sum_225: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [3], True);  mul_586 = None
    mul_587: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_583, sum_225);  sum_225 = None
    sub_180: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_585, sum_224);  mul_585 = sum_224 = None
    sub_181: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_180, mul_587);  sub_180 = mul_587 = None
    div_53: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 192);  rsqrt_10 = None
    mul_588: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_53, sub_181);  div_53 = sub_181 = None
    mul_589: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_673, mul_583);  mul_583 = None
    sum_226: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 1, 2]);  mul_589 = None
    sum_227: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_673, [0, 1, 2]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_251: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_560, mul_588);  permute_560 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_228: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1, 2], True)
    view_674: "f32[192]" = torch.ops.aten.view.default(sum_228, [192]);  sum_228 = None
    clone_260: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_251, memory_format = torch.contiguous_format)
    view_675: "f32[6272, 192]" = torch.ops.aten.view.default(clone_260, [6272, 192]);  clone_260 = None
    permute_569: "f32[192, 6272]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_167: "f32[192, 192]" = torch.ops.aten.mm.default(permute_569, view_66);  permute_569 = view_66 = None
    permute_570: "f32[192, 192]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    permute_571: "f32[192, 192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_168: "f32[6272, 192]" = torch.ops.aten.mm.default(view_675, permute_571);  view_675 = permute_571 = None
    view_676: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_168, [8, 28, 28, 192]);  mm_168 = None
    permute_572: "f32[192, 192]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    permute_573: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_676, [0, 3, 1, 2]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    iota_32: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_62: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
    iota_33: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_63: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
    add_252: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_62, unsqueeze_63);  unsqueeze_62 = unsqueeze_63 = None
    iota_34: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_64: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
    iota_35: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_65: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
    add_253: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_64, unsqueeze_65);  unsqueeze_64 = unsqueeze_65 = None
    constant_pad_nd_8: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_573, [1, 1, 1, 1], 0.0);  permute_573 = None
    unsqueeze_66: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_252, -1);  add_252 = None
    unsqueeze_67: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    slice_32: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, 0, 0, 9223372036854775807);  constant_pad_nd_8 = None
    slice_33: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 9223372036854775807);  slice_32 = None
    index_4: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_33, [None, None, unsqueeze_67, add_253]);  slice_33 = unsqueeze_67 = add_253 = None
    permute_574: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
    clone_261: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
    view_677: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_261, [8, 1728, 196]);  clone_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_678: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_677, [8, 6, 32, 9, 196]);  view_677 = None
    permute_575: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_678, [0, 1, 4, 3, 2]);  view_678 = None
    clone_262: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
    view_679: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_262, [9408, 9, 32]);  clone_262 = None
    permute_576: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_100: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_576, view_679);  permute_576 = None
    permute_577: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_101: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_679, permute_577);  view_679 = permute_577 = None
    view_680: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_100, [8, 6, 196, 9, 32]);  bmm_100 = None
    view_681: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_101, [8, 6, 196, 9, 9]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_39: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_590: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_681, alias_39);  view_681 = None
    sum_229: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [-1], True)
    mul_591: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_39, sum_229);  alias_39 = sum_229 = None
    sub_182: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_592: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_182, 0.1767766952966369);  sub_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_578: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_592, [0, 2, 1, 3, 4]);  mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_263: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
    view_682: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_263, [8, 14, 14, 486]);  clone_263 = None
    view_683: "f32[1568, 486]" = torch.ops.aten.view.default(view_682, [1568, 486]);  view_682 = None
    permute_579: "f32[486, 192]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_169: "f32[1568, 192]" = torch.ops.aten.mm.default(view_683, permute_579);  permute_579 = None
    permute_580: "f32[486, 1568]" = torch.ops.aten.permute.default(view_683, [1, 0])
    mm_170: "f32[486, 192]" = torch.ops.aten.mm.default(permute_580, view_58);  permute_580 = view_58 = None
    permute_581: "f32[192, 486]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_230: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True);  view_683 = None
    view_684: "f32[486]" = torch.ops.aten.view.default(sum_230, [486]);  sum_230 = None
    permute_582: "f32[486, 192]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_685: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_169, [8, 14, 14, 192]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_583: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_685, [0, 3, 1, 2]);  view_685 = None
    avg_pool2d_backward: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_583, permute_47, [2, 2], [2, 2], [0, 0], True, True, None);  permute_583 = permute_47 = None
    permute_584: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward, [0, 2, 3, 1]);  avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_585: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_680, [0, 1, 4, 3, 2]);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_264: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
    view_686: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_264, [8, 1728, 196]);  clone_264 = None
    view_687: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_686, [8, 192, 3, 3, 14, 14]);  view_686 = None
    permute_586: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_687, [0, 1, 2, 4, 3, 5]);  view_687 = None
    iota_36: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_68: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
    iota_37: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_69: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
    add_254: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
    unsqueeze_70: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_254, -1);  add_254 = None
    unsqueeze_71: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    iota_38: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_72: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
    iota_39: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_73: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
    add_255: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_72, unsqueeze_73);  unsqueeze_72 = unsqueeze_73 = None
    full_23: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_4: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_23, [None, None, unsqueeze_71, add_255], permute_586, True);  full_23 = unsqueeze_71 = add_255 = permute_586 = None
    constant_pad_nd_9: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_4, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_587: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_9, [0, 2, 3, 1]);  constant_pad_nd_9 = None
    clone_265: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_688: "f32[6272, 192]" = torch.ops.aten.view.default(clone_265, [6272, 192]);  clone_265 = None
    permute_588: "f32[192, 6272]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_171: "f32[192, 192]" = torch.ops.aten.mm.default(permute_588, view_54);  permute_588 = view_54 = None
    permute_589: "f32[192, 192]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    permute_590: "f32[192, 192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_172: "f32[6272, 192]" = torch.ops.aten.mm.default(view_688, permute_590);  view_688 = permute_590 = None
    view_689: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_172, [8, 28, 28, 192]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_256: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_584, view_689);  permute_584 = view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_591: "f32[192, 192]" = torch.ops.aten.permute.default(permute_589, [1, 0]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_266: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format);  add_50 = None
    sub_183: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_266, getitem_19);  clone_266 = getitem_19 = None
    mul_593: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_183, rsqrt_9);  sub_183 = None
    mul_594: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_256, primals_53);  primals_53 = None
    mul_595: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_594, 192)
    sum_231: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [3], True)
    mul_596: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_594, mul_593);  mul_594 = None
    sum_232: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [3], True);  mul_596 = None
    mul_597: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_593, sum_232);  sum_232 = None
    sub_184: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_595, sum_231);  mul_595 = sum_231 = None
    sub_185: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_184, mul_597);  sub_184 = mul_597 = None
    div_54: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 192);  rsqrt_9 = None
    mul_598: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_54, sub_185);  div_54 = sub_185 = None
    mul_599: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_256, mul_593);  mul_593 = None
    sum_233: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 1, 2]);  mul_599 = None
    sum_234: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_256, [0, 1, 2]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_257: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_251, mul_598);  add_251 = mul_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_267: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
    view_690: "f32[6272, 192]" = torch.ops.aten.view.default(clone_267, [6272, 192]);  clone_267 = None
    permute_592: "f32[192, 576]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_173: "f32[6272, 576]" = torch.ops.aten.mm.default(view_690, permute_592);  permute_592 = None
    permute_593: "f32[192, 6272]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_174: "f32[192, 576]" = torch.ops.aten.mm.default(permute_593, view_52);  permute_593 = view_52 = None
    permute_594: "f32[576, 192]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_235: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
    view_691: "f32[192]" = torch.ops.aten.view.default(sum_235, [192]);  sum_235 = None
    permute_595: "f32[192, 576]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_692: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_173, [8, 28, 28, 576]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_600: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_37: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_600);  mul_600 = None
    add_258: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_601: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_258, 0.5);  add_258 = None
    mul_602: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_603: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_602, -0.5);  mul_602 = None
    exp_37: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_603);  mul_603 = None
    mul_604: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_605: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, mul_604);  view_51 = mul_604 = None
    add_259: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_601, mul_605);  mul_601 = mul_605 = None
    mul_606: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_692, add_259);  view_692 = add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_693: "f32[6272, 576]" = torch.ops.aten.view.default(mul_606, [6272, 576]);  mul_606 = None
    permute_596: "f32[576, 192]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_175: "f32[6272, 192]" = torch.ops.aten.mm.default(view_693, permute_596);  permute_596 = None
    permute_597: "f32[576, 6272]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_176: "f32[576, 192]" = torch.ops.aten.mm.default(permute_597, view_50);  permute_597 = view_50 = None
    permute_598: "f32[192, 576]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_236: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
    view_694: "f32[576]" = torch.ops.aten.view.default(sum_236, [576]);  sum_236 = None
    permute_599: "f32[576, 192]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_695: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_175, [8, 28, 28, 192]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_268: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format);  add_46 = None
    sub_186: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_268, getitem_17);  clone_268 = getitem_17 = None
    mul_607: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_186, rsqrt_8);  sub_186 = None
    mul_608: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_695, primals_47);  primals_47 = None
    mul_609: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_608, 192)
    sum_237: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True)
    mul_610: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_608, mul_607);  mul_608 = None
    sum_238: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [3], True);  mul_610 = None
    mul_611: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_607, sum_238);  sum_238 = None
    sub_187: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_609, sum_237);  mul_609 = sum_237 = None
    sub_188: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_187, mul_611);  sub_187 = mul_611 = None
    div_55: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 192);  rsqrt_8 = None
    mul_612: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_55, sub_188);  div_55 = sub_188 = None
    mul_613: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_695, mul_607);  mul_607 = None
    sum_239: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1, 2]);  mul_613 = None
    sum_240: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_695, [0, 1, 2]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_260: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_257, mul_612);  add_257 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_241: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1, 2], True)
    view_696: "f32[192]" = torch.ops.aten.view.default(sum_241, [192]);  sum_241 = None
    clone_269: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
    view_697: "f32[6272, 192]" = torch.ops.aten.view.default(clone_269, [6272, 192]);  clone_269 = None
    permute_600: "f32[192, 6272]" = torch.ops.aten.permute.default(view_697, [1, 0])
    mm_177: "f32[192, 192]" = torch.ops.aten.mm.default(permute_600, view_48);  permute_600 = view_48 = None
    permute_601: "f32[192, 192]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    permute_602: "f32[192, 192]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_178: "f32[6272, 192]" = torch.ops.aten.mm.default(view_697, permute_602);  view_697 = permute_602 = None
    view_698: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_178, [8, 28, 28, 192]);  mm_178 = None
    permute_603: "f32[192, 192]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    permute_604: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_698, [0, 3, 1, 2]);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    iota_40: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_74: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
    iota_41: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_75: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
    add_261: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_74, unsqueeze_75);  unsqueeze_74 = unsqueeze_75 = None
    iota_42: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_76: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
    iota_43: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_77: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
    add_262: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_76, unsqueeze_77);  unsqueeze_76 = unsqueeze_77 = None
    constant_pad_nd_10: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_604, [1, 1, 1, 1], 0.0);  permute_604 = None
    unsqueeze_78: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_261, -1);  add_261 = None
    unsqueeze_79: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    slice_34: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, 0, 0, 9223372036854775807);  constant_pad_nd_10 = None
    slice_35: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_34, 1, 0, 9223372036854775807);  slice_34 = None
    index_5: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_35, [None, None, unsqueeze_79, add_262]);  slice_35 = unsqueeze_79 = add_262 = None
    permute_605: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
    clone_270: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_605, memory_format = torch.contiguous_format);  permute_605 = None
    view_699: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_270, [8, 1728, 196]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_700: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_699, [8, 6, 32, 9, 196]);  view_699 = None
    permute_606: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_700, [0, 1, 4, 3, 2]);  view_700 = None
    clone_271: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_606, memory_format = torch.contiguous_format);  permute_606 = None
    view_701: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_271, [9408, 9, 32]);  clone_271 = None
    permute_607: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    bmm_102: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_607, view_701);  permute_607 = None
    permute_608: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_103: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_701, permute_608);  view_701 = permute_608 = None
    view_702: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_102, [8, 6, 196, 9, 32]);  bmm_102 = None
    view_703: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_103, [8, 6, 196, 9, 9]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_40: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_614: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_703, alias_40);  view_703 = None
    sum_242: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
    mul_615: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_40, sum_242);  alias_40 = sum_242 = None
    sub_189: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_616: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_189, 0.1767766952966369);  sub_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_609: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_616, [0, 2, 1, 3, 4]);  mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_272: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    view_704: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_272, [8, 14, 14, 486]);  clone_272 = None
    view_705: "f32[1568, 486]" = torch.ops.aten.view.default(view_704, [1568, 486]);  view_704 = None
    permute_610: "f32[486, 192]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_179: "f32[1568, 192]" = torch.ops.aten.mm.default(view_705, permute_610);  permute_610 = None
    permute_611: "f32[486, 1568]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_180: "f32[486, 192]" = torch.ops.aten.mm.default(permute_611, view_40);  permute_611 = view_40 = None
    permute_612: "f32[192, 486]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_243: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[486]" = torch.ops.aten.view.default(sum_243, [486]);  sum_243 = None
    permute_613: "f32[486, 192]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_707: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_179, [8, 14, 14, 192]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_614: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_707, [0, 3, 1, 2]);  view_707 = None
    avg_pool2d_backward_1: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_614, permute_33, [2, 2], [2, 2], [0, 0], True, True, None);  permute_614 = permute_33 = None
    permute_615: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_1, [0, 2, 3, 1]);  avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_616: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_702, [0, 1, 4, 3, 2]);  view_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_273: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None
    view_708: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_273, [8, 1728, 196]);  clone_273 = None
    view_709: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_708, [8, 192, 3, 3, 14, 14]);  view_708 = None
    permute_617: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_709, [0, 1, 2, 4, 3, 5]);  view_709 = None
    iota_44: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_80: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
    iota_45: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_81: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
    add_263: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_80, unsqueeze_81);  unsqueeze_80 = unsqueeze_81 = None
    unsqueeze_82: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_263, -1);  add_263 = None
    unsqueeze_83: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    iota_46: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_84: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
    iota_47: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_85: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
    add_264: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_84, unsqueeze_85);  unsqueeze_84 = unsqueeze_85 = None
    full_24: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_5: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_24, [None, None, unsqueeze_83, add_264], permute_617, True);  full_24 = unsqueeze_83 = add_264 = permute_617 = None
    constant_pad_nd_11: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_5, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_618: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_11, [0, 2, 3, 1]);  constant_pad_nd_11 = None
    clone_274: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_618, memory_format = torch.contiguous_format);  permute_618 = None
    view_710: "f32[6272, 192]" = torch.ops.aten.view.default(clone_274, [6272, 192]);  clone_274 = None
    permute_619: "f32[192, 6272]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_181: "f32[192, 192]" = torch.ops.aten.mm.default(permute_619, view_36);  permute_619 = view_36 = None
    permute_620: "f32[192, 192]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    permute_621: "f32[192, 192]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_182: "f32[6272, 192]" = torch.ops.aten.mm.default(view_710, permute_621);  view_710 = permute_621 = None
    view_711: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_182, [8, 28, 28, 192]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_265: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_615, view_711);  permute_615 = view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_622: "f32[192, 192]" = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_275: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
    sub_190: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_275, getitem_15);  clone_275 = getitem_15 = None
    mul_617: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_7);  sub_190 = None
    mul_618: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_265, primals_40);  primals_40 = None
    mul_619: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_618, 192)
    sum_244: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [3], True)
    mul_620: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_618, mul_617);  mul_618 = None
    sum_245: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True);  mul_620 = None
    mul_621: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_617, sum_245);  sum_245 = None
    sub_191: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_619, sum_244);  mul_619 = sum_244 = None
    sub_192: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_191, mul_621);  sub_191 = mul_621 = None
    div_56: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 192);  rsqrt_7 = None
    mul_622: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_56, sub_192);  div_56 = sub_192 = None
    mul_623: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_265, mul_617);  mul_617 = None
    sum_246: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1, 2]);  mul_623 = None
    sum_247: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_265, [0, 1, 2]);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_266: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_260, mul_622);  add_260 = mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_276: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_266, memory_format = torch.contiguous_format)
    view_712: "f32[6272, 192]" = torch.ops.aten.view.default(clone_276, [6272, 192]);  clone_276 = None
    permute_623: "f32[192, 576]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_183: "f32[6272, 576]" = torch.ops.aten.mm.default(view_712, permute_623);  permute_623 = None
    permute_624: "f32[192, 6272]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_184: "f32[192, 576]" = torch.ops.aten.mm.default(permute_624, view_34);  permute_624 = view_34 = None
    permute_625: "f32[576, 192]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_248: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[192]" = torch.ops.aten.view.default(sum_248, [192]);  sum_248 = None
    permute_626: "f32[192, 576]" = torch.ops.aten.permute.default(permute_625, [1, 0]);  permute_625 = None
    view_714: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_183, [8, 28, 28, 576]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_624: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_38: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_624);  mul_624 = None
    add_267: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_625: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_267, 0.5);  add_267 = None
    mul_626: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_627: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_626, -0.5);  mul_626 = None
    exp_38: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_627);  mul_627 = None
    mul_628: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_629: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, mul_628);  view_33 = mul_628 = None
    add_268: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_625, mul_629);  mul_625 = mul_629 = None
    mul_630: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_714, add_268);  view_714 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_715: "f32[6272, 576]" = torch.ops.aten.view.default(mul_630, [6272, 576]);  mul_630 = None
    permute_627: "f32[576, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_185: "f32[6272, 192]" = torch.ops.aten.mm.default(view_715, permute_627);  permute_627 = None
    permute_628: "f32[576, 6272]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_186: "f32[576, 192]" = torch.ops.aten.mm.default(permute_628, view_32);  permute_628 = view_32 = None
    permute_629: "f32[192, 576]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_249: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[576]" = torch.ops.aten.view.default(sum_249, [576]);  sum_249 = None
    permute_630: "f32[576, 192]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
    view_717: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_185, [8, 28, 28, 192]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_277: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format);  add_34 = None
    sub_193: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_277, getitem_13);  clone_277 = getitem_13 = None
    mul_631: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_6);  sub_193 = None
    mul_632: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_717, primals_34);  primals_34 = None
    mul_633: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_632, 192)
    sum_250: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [3], True)
    mul_634: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_632, mul_631);  mul_632 = None
    sum_251: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [3], True);  mul_634 = None
    mul_635: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_631, sum_251);  sum_251 = None
    sub_194: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_633, sum_250);  mul_633 = sum_250 = None
    sub_195: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_194, mul_635);  sub_194 = mul_635 = None
    div_57: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 192);  rsqrt_6 = None
    mul_636: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_57, sub_195);  div_57 = sub_195 = None
    mul_637: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_717, mul_631);  mul_631 = None
    sum_252: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1, 2]);  mul_637 = None
    sum_253: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 1, 2]);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_269: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_266, mul_636);  add_266 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_254: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1, 2], True)
    view_718: "f32[192]" = torch.ops.aten.view.default(sum_254, [192]);  sum_254 = None
    clone_278: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_269, memory_format = torch.contiguous_format)
    view_719: "f32[6272, 192]" = torch.ops.aten.view.default(clone_278, [6272, 192]);  clone_278 = None
    permute_631: "f32[192, 6272]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_187: "f32[192, 192]" = torch.ops.aten.mm.default(permute_631, view_30);  permute_631 = view_30 = None
    permute_632: "f32[192, 192]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    permute_633: "f32[192, 192]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_188: "f32[6272, 192]" = torch.ops.aten.mm.default(view_719, permute_633);  view_719 = permute_633 = None
    view_720: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_188, [8, 28, 28, 192]);  mm_188 = None
    permute_634: "f32[192, 192]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    permute_635: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_720, [0, 3, 1, 2]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    iota_48: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_86: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_48, 0);  iota_48 = None
    iota_49: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_87: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_49, -1);  iota_49 = None
    add_270: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_86, unsqueeze_87);  unsqueeze_86 = unsqueeze_87 = None
    iota_50: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_88: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_50, 0);  iota_50 = None
    iota_51: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_89: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_51, -1);  iota_51 = None
    add_271: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_88, unsqueeze_89);  unsqueeze_88 = unsqueeze_89 = None
    constant_pad_nd_12: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_635, [1, 1, 1, 1], 0.0);  permute_635 = None
    unsqueeze_90: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_270, -1);  add_270 = None
    unsqueeze_91: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    slice_36: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_12, 0, 0, 9223372036854775807);  constant_pad_nd_12 = None
    slice_37: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_36, 1, 0, 9223372036854775807);  slice_36 = None
    index_6: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_37, [None, None, unsqueeze_91, add_271]);  slice_37 = unsqueeze_91 = add_271 = None
    permute_636: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
    clone_279: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
    view_721: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_279, [8, 1728, 196]);  clone_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_722: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_721, [8, 6, 32, 9, 196]);  view_721 = None
    permute_637: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_722, [0, 1, 4, 3, 2]);  view_722 = None
    clone_280: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format);  permute_637 = None
    view_723: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_280, [9408, 9, 32]);  clone_280 = None
    permute_638: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_104: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_638, view_723);  permute_638 = None
    permute_639: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_105: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_723, permute_639);  view_723 = permute_639 = None
    view_724: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_104, [8, 6, 196, 9, 32]);  bmm_104 = None
    view_725: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_105, [8, 6, 196, 9, 9]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_41: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_638: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_725, alias_41);  view_725 = None
    sum_255: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [-1], True)
    mul_639: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_41, sum_255);  alias_41 = sum_255 = None
    sub_196: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_640: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_196, 0.1767766952966369);  sub_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_640: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_640, [0, 2, 1, 3, 4]);  mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_281: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_640, memory_format = torch.contiguous_format);  permute_640 = None
    view_726: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_281, [8, 14, 14, 486]);  clone_281 = None
    view_727: "f32[1568, 486]" = torch.ops.aten.view.default(view_726, [1568, 486]);  view_726 = None
    permute_641: "f32[486, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_189: "f32[1568, 192]" = torch.ops.aten.mm.default(view_727, permute_641);  permute_641 = None
    permute_642: "f32[486, 1568]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_190: "f32[486, 192]" = torch.ops.aten.mm.default(permute_642, view_22);  permute_642 = view_22 = None
    permute_643: "f32[192, 486]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_256: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[486]" = torch.ops.aten.view.default(sum_256, [486]);  sum_256 = None
    permute_644: "f32[486, 192]" = torch.ops.aten.permute.default(permute_643, [1, 0]);  permute_643 = None
    view_729: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_189, [8, 14, 14, 192]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_645: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_729, [0, 3, 1, 2]);  view_729 = None
    avg_pool2d_backward_2: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_645, permute_19, [2, 2], [2, 2], [0, 0], True, True, None);  permute_645 = permute_19 = None
    permute_646: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_2, [0, 2, 3, 1]);  avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_647: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_724, [0, 1, 4, 3, 2]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_282: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_647, memory_format = torch.contiguous_format);  permute_647 = None
    view_730: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_282, [8, 1728, 196]);  clone_282 = None
    view_731: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_730, [8, 192, 3, 3, 14, 14]);  view_730 = None
    permute_648: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_731, [0, 1, 2, 4, 3, 5]);  view_731 = None
    iota_52: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_92: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_52, 0);  iota_52 = None
    iota_53: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_93: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_53, -1);  iota_53 = None
    add_272: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_92, unsqueeze_93);  unsqueeze_92 = unsqueeze_93 = None
    unsqueeze_94: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_272, -1);  add_272 = None
    unsqueeze_95: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    iota_54: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_96: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_54, 0);  iota_54 = None
    iota_55: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_97: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_55, -1);  iota_55 = None
    add_273: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_96, unsqueeze_97);  unsqueeze_96 = unsqueeze_97 = None
    full_25: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_6: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_25, [None, None, unsqueeze_95, add_273], permute_648, True);  full_25 = unsqueeze_95 = add_273 = permute_648 = None
    constant_pad_nd_13: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_6, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_649: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_13, [0, 2, 3, 1]);  constant_pad_nd_13 = None
    clone_283: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None
    view_732: "f32[6272, 192]" = torch.ops.aten.view.default(clone_283, [6272, 192]);  clone_283 = None
    permute_650: "f32[192, 6272]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_191: "f32[192, 192]" = torch.ops.aten.mm.default(permute_650, view_18);  permute_650 = view_18 = None
    permute_651: "f32[192, 192]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    permute_652: "f32[192, 192]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_192: "f32[6272, 192]" = torch.ops.aten.mm.default(view_732, permute_652);  view_732 = permute_652 = None
    view_733: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_192, [8, 28, 28, 192]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_274: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_646, view_733);  permute_646 = view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_653: "f32[192, 192]" = torch.ops.aten.permute.default(permute_651, [1, 0]);  permute_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_284: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format);  add_26 = None
    sub_197: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_284, getitem_11);  clone_284 = getitem_11 = None
    mul_641: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_5);  sub_197 = None
    mul_642: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_274, primals_27);  primals_27 = None
    mul_643: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_642, 192)
    sum_257: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [3], True)
    mul_644: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_642, mul_641);  mul_642 = None
    sum_258: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [3], True);  mul_644 = None
    mul_645: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_641, sum_258);  sum_258 = None
    sub_198: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_643, sum_257);  mul_643 = sum_257 = None
    sub_199: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_198, mul_645);  sub_198 = mul_645 = None
    div_58: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 192);  rsqrt_5 = None
    mul_646: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_58, sub_199);  div_58 = sub_199 = None
    mul_647: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_274, mul_641);  mul_641 = None
    sum_259: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1, 2]);  mul_647 = None
    sum_260: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1, 2]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_275: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_269, mul_646);  add_269 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_285: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
    view_734: "f32[6272, 192]" = torch.ops.aten.view.default(clone_285, [6272, 192]);  clone_285 = None
    permute_654: "f32[192, 576]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_193: "f32[6272, 576]" = torch.ops.aten.mm.default(view_734, permute_654);  permute_654 = None
    permute_655: "f32[192, 6272]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_194: "f32[192, 576]" = torch.ops.aten.mm.default(permute_655, view_16);  permute_655 = view_16 = None
    permute_656: "f32[576, 192]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_261: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[192]" = torch.ops.aten.view.default(sum_261, [192]);  sum_261 = None
    permute_657: "f32[192, 576]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    view_736: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_193, [8, 28, 28, 576]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_648: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf_39: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_648);  mul_648 = None
    add_276: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_649: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_276, 0.5);  add_276 = None
    mul_650: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_651: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_650, -0.5);  mul_650 = None
    exp_39: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_651);  mul_651 = None
    mul_652: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_653: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, mul_652);  view_15 = mul_652 = None
    add_277: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_649, mul_653);  mul_649 = mul_653 = None
    mul_654: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_736, add_277);  view_736 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_737: "f32[6272, 576]" = torch.ops.aten.view.default(mul_654, [6272, 576]);  mul_654 = None
    permute_658: "f32[576, 192]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_195: "f32[6272, 192]" = torch.ops.aten.mm.default(view_737, permute_658);  permute_658 = None
    permute_659: "f32[576, 6272]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_196: "f32[576, 192]" = torch.ops.aten.mm.default(permute_659, view_14);  permute_659 = view_14 = None
    permute_660: "f32[192, 576]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_262: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[576]" = torch.ops.aten.view.default(sum_262, [576]);  sum_262 = None
    permute_661: "f32[576, 192]" = torch.ops.aten.permute.default(permute_660, [1, 0]);  permute_660 = None
    view_739: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_195, [8, 28, 28, 192]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_286: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format);  add_22 = None
    sub_200: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_286, getitem_9);  clone_286 = getitem_9 = None
    mul_655: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_4);  sub_200 = None
    mul_656: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_739, primals_21);  primals_21 = None
    mul_657: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_656, 192)
    sum_263: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [3], True)
    mul_658: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_656, mul_655);  mul_656 = None
    sum_264: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [3], True);  mul_658 = None
    mul_659: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_655, sum_264);  sum_264 = None
    sub_201: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_657, sum_263);  mul_657 = sum_263 = None
    sub_202: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_201, mul_659);  sub_201 = mul_659 = None
    div_59: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 192);  rsqrt_4 = None
    mul_660: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_59, sub_202);  div_59 = sub_202 = None
    mul_661: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_739, mul_655);  mul_655 = None
    sum_265: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1, 2]);  mul_661 = None
    sum_266: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_739, [0, 1, 2]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_278: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_275, mul_660);  add_275 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_267: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1, 2], True)
    view_740: "f32[192]" = torch.ops.aten.view.default(sum_267, [192]);  sum_267 = None
    clone_287: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
    view_741: "f32[6272, 192]" = torch.ops.aten.view.default(clone_287, [6272, 192]);  clone_287 = None
    permute_662: "f32[192, 6272]" = torch.ops.aten.permute.default(view_741, [1, 0])
    mm_197: "f32[192, 192]" = torch.ops.aten.mm.default(permute_662, view_12);  permute_662 = view_12 = None
    permute_663: "f32[192, 192]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    permute_664: "f32[192, 192]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_198: "f32[6272, 192]" = torch.ops.aten.mm.default(view_741, permute_664);  view_741 = permute_664 = None
    view_742: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_198, [8, 28, 28, 192]);  mm_198 = None
    permute_665: "f32[192, 192]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    permute_666: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_742, [0, 3, 1, 2]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    iota_56: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_98: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_56, 0);  iota_56 = None
    iota_57: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_99: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_57, -1);  iota_57 = None
    add_279: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
    iota_58: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_100: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_58, 0);  iota_58 = None
    iota_59: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_101: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_59, -1);  iota_59 = None
    add_280: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_100, unsqueeze_101);  unsqueeze_100 = unsqueeze_101 = None
    constant_pad_nd_14: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_666, [1, 1, 1, 1], 0.0);  permute_666 = None
    unsqueeze_102: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_279, -1);  add_279 = None
    unsqueeze_103: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    slice_38: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_14, 0, 0, 9223372036854775807);  constant_pad_nd_14 = None
    slice_39: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 9223372036854775807);  slice_38 = None
    index_7: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_39, [None, None, unsqueeze_103, add_280]);  slice_39 = unsqueeze_103 = add_280 = None
    permute_667: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
    clone_288: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
    view_743: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_288, [8, 1728, 196]);  clone_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_744: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_743, [8, 6, 32, 9, 196]);  view_743 = None
    permute_668: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_744, [0, 1, 4, 3, 2]);  view_744 = None
    clone_289: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_668, memory_format = torch.contiguous_format);  permute_668 = None
    view_745: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_289, [9408, 9, 32]);  clone_289 = None
    permute_669: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_106: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_669, view_745);  permute_669 = None
    permute_670: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    bmm_107: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_745, permute_670);  view_745 = permute_670 = None
    view_746: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_106, [8, 6, 196, 9, 32]);  bmm_106 = None
    view_747: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_107, [8, 6, 196, 9, 9]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_662: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_747, alias_42);  view_747 = None
    sum_268: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_662, [-1], True)
    mul_663: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_42, sum_268);  alias_42 = sum_268 = None
    sub_203: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_664: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_203, 0.1767766952966369);  sub_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_671: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_664, [0, 2, 1, 3, 4]);  mul_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_290: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
    view_748: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_290, [8, 14, 14, 486]);  clone_290 = None
    view_749: "f32[1568, 486]" = torch.ops.aten.view.default(view_748, [1568, 486]);  view_748 = None
    permute_672: "f32[486, 192]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_199: "f32[1568, 192]" = torch.ops.aten.mm.default(view_749, permute_672);  permute_672 = None
    permute_673: "f32[486, 1568]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_200: "f32[486, 192]" = torch.ops.aten.mm.default(permute_673, view_4);  permute_673 = view_4 = None
    permute_674: "f32[192, 486]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_269: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[486]" = torch.ops.aten.view.default(sum_269, [486]);  sum_269 = None
    permute_675: "f32[486, 192]" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
    view_751: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_199, [8, 14, 14, 192]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_676: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_751, [0, 3, 1, 2]);  view_751 = None
    avg_pool2d_backward_3: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_676, permute_5, [2, 2], [2, 2], [0, 0], True, True, None);  permute_676 = permute_5 = None
    permute_677: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_3, [0, 2, 3, 1]);  avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_678: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_746, [0, 1, 4, 3, 2]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_291: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
    view_752: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_291, [8, 1728, 196]);  clone_291 = None
    view_753: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_752, [8, 192, 3, 3, 14, 14]);  view_752 = None
    permute_679: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_753, [0, 1, 2, 4, 3, 5]);  view_753 = None
    iota_60: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_104: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_60, 0);  iota_60 = None
    iota_61: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_105: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_61, -1);  iota_61 = None
    add_281: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_104, unsqueeze_105);  unsqueeze_104 = unsqueeze_105 = None
    unsqueeze_106: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_281, -1);  add_281 = None
    unsqueeze_107: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    iota_62: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_108: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_62, 0);  iota_62 = None
    iota_63: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_109: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_63, -1);  iota_63 = None
    add_282: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_108, unsqueeze_109);  unsqueeze_108 = unsqueeze_109 = None
    full_26: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_7: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_26, [None, None, unsqueeze_107, add_282], permute_679, True);  full_26 = unsqueeze_107 = add_282 = permute_679 = None
    constant_pad_nd_15: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_7, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_680: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_15, [0, 2, 3, 1]);  constant_pad_nd_15 = None
    clone_292: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_680, memory_format = torch.contiguous_format);  permute_680 = None
    view_754: "f32[6272, 192]" = torch.ops.aten.view.default(clone_292, [6272, 192]);  clone_292 = None
    permute_681: "f32[192, 6272]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_201: "f32[192, 192]" = torch.ops.aten.mm.default(permute_681, view);  permute_681 = view = None
    permute_682: "f32[192, 192]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    permute_683: "f32[192, 192]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_202: "f32[6272, 192]" = torch.ops.aten.mm.default(view_754, permute_683);  view_754 = permute_683 = None
    view_755: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_202, [8, 28, 28, 192]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_283: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_677, view_755);  permute_677 = view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_684: "f32[192, 192]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_293: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_204: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_293, getitem_7);  clone_293 = getitem_7 = None
    mul_665: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_204, rsqrt_3);  sub_204 = None
    mul_666: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_283, primals_14);  primals_14 = None
    mul_667: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_666, 192)
    sum_270: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [3], True)
    mul_668: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_666, mul_665);  mul_666 = None
    sum_271: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [3], True);  mul_668 = None
    mul_669: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_665, sum_271);  sum_271 = None
    sub_205: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_667, sum_270);  mul_667 = sum_270 = None
    sub_206: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_205, mul_669);  sub_205 = mul_669 = None
    div_60: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 192);  rsqrt_3 = None
    mul_670: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_60, sub_206);  div_60 = sub_206 = None
    mul_671: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_283, mul_665);  mul_665 = None
    sum_272: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1, 2]);  mul_671 = None
    sum_273: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_283, [0, 1, 2]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_284: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_278, mul_670);  add_278 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    permute_685: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_284, [0, 3, 1, 2]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    sum_274: "f32[192]" = torch.ops.aten.sum.dim_IntList(permute_685, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_685, relu_2, primals_12, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_685 = primals_12 = None
    getitem_139: "f32[8, 64, 112, 112]" = convolution_backward_1[0]
    getitem_140: "f32[192, 64, 4, 4]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    alias_44: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_45: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le, scalar_tensor, getitem_139);  le = scalar_tensor = getitem_139 = None
    unsqueeze_110: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_111: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
    unsqueeze_112: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
    sum_275: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_207: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_112)
    mul_672: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where, sub_207);  sub_207 = None
    sum_276: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
    mul_673: "f32[64]" = torch.ops.aten.mul.Tensor(sum_275, 9.964923469387754e-06)
    unsqueeze_113: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_114: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    unsqueeze_115: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
    mul_674: "f32[64]" = torch.ops.aten.mul.Tensor(sum_276, 9.964923469387754e-06)
    mul_675: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_676: "f32[64]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_116: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_117: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
    unsqueeze_118: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
    mul_677: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_10);  primals_10 = None
    unsqueeze_119: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_120: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    unsqueeze_121: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
    sub_208: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_112);  convolution_2 = unsqueeze_112 = None
    mul_678: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_118);  sub_208 = unsqueeze_118 = None
    sub_209: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where, mul_678);  where = mul_678 = None
    sub_210: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_115);  sub_209 = unsqueeze_115 = None
    mul_679: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_121);  sub_210 = unsqueeze_121 = None
    mul_680: "f32[64]" = torch.ops.aten.mul.Tensor(sum_276, squeeze_7);  sum_276 = squeeze_7 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_679, relu_1, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_679 = primals_9 = None
    getitem_142: "f32[8, 64, 112, 112]" = convolution_backward_2[0]
    getitem_143: "f32[64, 64, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    alias_47: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_48: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_1: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_142);  le_1 = scalar_tensor_1 = getitem_142 = None
    unsqueeze_122: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_123: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
    unsqueeze_124: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
    sum_277: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_211: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_124)
    mul_681: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_1, sub_211);  sub_211 = None
    sum_278: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3]);  mul_681 = None
    mul_682: "f32[64]" = torch.ops.aten.mul.Tensor(sum_277, 9.964923469387754e-06)
    unsqueeze_125: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_126: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    unsqueeze_127: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
    mul_683: "f32[64]" = torch.ops.aten.mul.Tensor(sum_278, 9.964923469387754e-06)
    mul_684: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_685: "f32[64]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_128: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_129: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    mul_686: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_7);  primals_7 = None
    unsqueeze_131: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_132: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    sub_212: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_124);  convolution_1 = unsqueeze_124 = None
    mul_687: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_130);  sub_212 = unsqueeze_130 = None
    sub_213: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_1, mul_687);  where_1 = mul_687 = None
    sub_214: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_127);  sub_213 = unsqueeze_127 = None
    mul_688: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_133);  sub_214 = unsqueeze_133 = None
    mul_689: "f32[64]" = torch.ops.aten.mul.Tensor(sum_278, squeeze_4);  sum_278 = squeeze_4 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_688, relu, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_688 = primals_6 = None
    getitem_145: "f32[8, 64, 112, 112]" = convolution_backward_3[0]
    getitem_146: "f32[64, 64, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    alias_50: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_51: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_2: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_145);  le_2 = scalar_tensor_2 = getitem_145 = None
    unsqueeze_134: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_135: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    sum_279: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_215: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_136)
    mul_690: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_2, sub_215);  sub_215 = None
    sum_280: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2, 3]);  mul_690 = None
    mul_691: "f32[64]" = torch.ops.aten.mul.Tensor(sum_279, 9.964923469387754e-06)
    unsqueeze_137: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_138: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    mul_692: "f32[64]" = torch.ops.aten.mul.Tensor(sum_280, 9.964923469387754e-06)
    mul_693: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_694: "f32[64]" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    unsqueeze_140: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_141: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    mul_695: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_4);  primals_4 = None
    unsqueeze_143: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_144: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    sub_216: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_136);  convolution = unsqueeze_136 = None
    mul_696: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_142);  sub_216 = unsqueeze_142 = None
    sub_217: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_2, mul_696);  where_2 = mul_696 = None
    sub_218: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_139);  sub_217 = unsqueeze_139 = None
    mul_697: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_145);  sub_218 = unsqueeze_145 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(sum_280, squeeze_1);  sum_280 = squeeze_1 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_697, primals_261, primals_3, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_697 = primals_261 = primals_3 = None
    getitem_149: "f32[64, 3, 7, 7]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_252, add_2);  primals_252 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_253, add_3);  primals_253 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_254, add);  primals_254 = add = None
    copy__3: "f32[64]" = torch.ops.aten.copy_.default(primals_255, add_7);  primals_255 = add_7 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_256, add_8);  primals_256 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_5);  primals_257 = add_5 = None
    copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_258, add_12);  primals_258 = add_12 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_259, add_13);  primals_259 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_10);  primals_260 = add_10 = None
    return pytree.tree_unflatten([add_179, sum_220, sum_51, getitem_149, mul_698, sum_279, getitem_146, mul_689, sum_277, getitem_143, mul_680, sum_275, getitem_140, sum_274, sum_272, sum_273, permute_684, permute_675, view_750, permute_665, view_740, sum_265, sum_266, permute_661, view_738, permute_657, view_735, sum_259, sum_260, permute_653, permute_644, view_728, permute_634, view_718, sum_252, sum_253, permute_630, view_716, permute_626, view_713, sum_246, sum_247, permute_622, permute_613, view_706, permute_603, view_696, sum_239, sum_240, permute_599, view_694, permute_595, view_691, sum_233, sum_234, permute_591, permute_582, view_684, permute_572, view_674, sum_226, sum_227, permute_568, view_672, permute_564, view_669, getitem_137, sum_221, sum_218, sum_219, permute_558, permute_547, view_655, sum_212, sum_213, permute_543, view_652, permute_539, view_649, sum_206, sum_207, permute_535, permute_524, view_635, sum_200, sum_201, permute_520, view_632, permute_516, view_629, sum_194, sum_195, permute_512, permute_501, view_615, sum_188, sum_189, permute_497, view_612, permute_493, view_609, sum_182, sum_183, permute_489, permute_478, view_595, sum_176, sum_177, permute_474, view_592, permute_470, view_589, sum_170, sum_171, permute_466, permute_455, view_575, sum_164, sum_165, permute_451, view_572, permute_447, view_569, sum_158, sum_159, permute_443, permute_432, view_555, sum_152, sum_153, permute_428, view_552, permute_424, view_549, sum_146, sum_147, permute_420, permute_409, view_535, sum_140, sum_141, permute_405, view_532, permute_401, view_529, sum_134, sum_135, permute_397, permute_386, view_515, sum_128, sum_129, permute_382, view_512, permute_378, view_509, sum_122, sum_123, permute_374, permute_363, view_495, sum_116, sum_117, permute_359, view_492, permute_355, view_489, sum_110, sum_111, permute_351, permute_340, view_475, sum_104, sum_105, permute_336, view_472, permute_332, view_469, sum_98, sum_99, permute_328, permute_317, view_455, sum_92, sum_93, permute_313, view_452, permute_309, view_449, sum_86, sum_87, permute_305, permute_294, view_435, sum_80, sum_81, permute_290, view_432, permute_286, view_429, sum_74, sum_75, permute_282, permute_271, view_415, sum_68, sum_69, permute_267, view_412, permute_263, view_409, sum_62, sum_63, permute_259, permute_248, view_395, sum_56, sum_57, permute_244, view_392, permute_240, view_389, sum_49, sum_50, permute_236, permute_231, permute_221, view_371, sum_43, sum_44, permute_217, view_368, permute_213, view_365, sum_37, sum_38, permute_209, permute_204, permute_194, view_348, sum_31, sum_32, permute_190, view_345, permute_186, view_342, sum_25, sum_26, permute_182, view_340, permute_178, view_337, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    