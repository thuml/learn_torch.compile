from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16, 3, 3, 3]"; primals_2: "f32[16]"; primals_3: "f32[16]"; primals_4: "f32[16, 1, 3, 3]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[16, 16, 1, 1]"; primals_8: "f32[16]"; primals_9: "f32[16]"; primals_10: "f32[64, 16, 1, 1]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[64, 1, 3, 3]"; primals_14: "f32[64]"; primals_15: "f32[64]"; primals_16: "f32[24, 64, 1, 1]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[72, 24, 1, 1]"; primals_20: "f32[72]"; primals_21: "f32[72]"; primals_22: "f32[72, 1, 3, 3]"; primals_23: "f32[72]"; primals_24: "f32[72]"; primals_25: "f32[24, 72, 1, 1]"; primals_26: "f32[24]"; primals_27: "f32[24]"; primals_28: "f32[72, 24, 1, 1]"; primals_29: "f32[72]"; primals_30: "f32[72]"; primals_31: "f32[72, 1, 5, 5]"; primals_32: "f32[72]"; primals_33: "f32[72]"; primals_34: "f32[24, 72, 1, 1]"; primals_35: "f32[24]"; primals_36: "f32[72, 24, 1, 1]"; primals_37: "f32[72]"; primals_38: "f32[40, 72, 1, 1]"; primals_39: "f32[40]"; primals_40: "f32[40]"; primals_41: "f32[120, 40, 1, 1]"; primals_42: "f32[120]"; primals_43: "f32[120]"; primals_44: "f32[120, 1, 5, 5]"; primals_45: "f32[120]"; primals_46: "f32[120]"; primals_47: "f32[32, 120, 1, 1]"; primals_48: "f32[32]"; primals_49: "f32[120, 32, 1, 1]"; primals_50: "f32[120]"; primals_51: "f32[40, 120, 1, 1]"; primals_52: "f32[40]"; primals_53: "f32[40]"; primals_54: "f32[120, 40, 1, 1]"; primals_55: "f32[120]"; primals_56: "f32[120]"; primals_57: "f32[120, 1, 5, 5]"; primals_58: "f32[120]"; primals_59: "f32[120]"; primals_60: "f32[32, 120, 1, 1]"; primals_61: "f32[32]"; primals_62: "f32[120, 32, 1, 1]"; primals_63: "f32[120]"; primals_64: "f32[40, 120, 1, 1]"; primals_65: "f32[40]"; primals_66: "f32[40]"; primals_67: "f32[240, 40, 1, 1]"; primals_68: "f32[240]"; primals_69: "f32[240]"; primals_70: "f32[240, 1, 3, 3]"; primals_71: "f32[240]"; primals_72: "f32[240]"; primals_73: "f32[80, 240, 1, 1]"; primals_74: "f32[80]"; primals_75: "f32[80]"; primals_76: "f32[200, 80, 1, 1]"; primals_77: "f32[200]"; primals_78: "f32[200]"; primals_79: "f32[200, 1, 3, 3]"; primals_80: "f32[200]"; primals_81: "f32[200]"; primals_82: "f32[80, 200, 1, 1]"; primals_83: "f32[80]"; primals_84: "f32[80]"; primals_85: "f32[184, 80, 1, 1]"; primals_86: "f32[184]"; primals_87: "f32[184]"; primals_88: "f32[184, 1, 3, 3]"; primals_89: "f32[184]"; primals_90: "f32[184]"; primals_91: "f32[80, 184, 1, 1]"; primals_92: "f32[80]"; primals_93: "f32[80]"; primals_94: "f32[184, 80, 1, 1]"; primals_95: "f32[184]"; primals_96: "f32[184]"; primals_97: "f32[184, 1, 3, 3]"; primals_98: "f32[184]"; primals_99: "f32[184]"; primals_100: "f32[80, 184, 1, 1]"; primals_101: "f32[80]"; primals_102: "f32[80]"; primals_103: "f32[480, 80, 1, 1]"; primals_104: "f32[480]"; primals_105: "f32[480]"; primals_106: "f32[480, 1, 3, 3]"; primals_107: "f32[480]"; primals_108: "f32[480]"; primals_109: "f32[120, 480, 1, 1]"; primals_110: "f32[120]"; primals_111: "f32[480, 120, 1, 1]"; primals_112: "f32[480]"; primals_113: "f32[112, 480, 1, 1]"; primals_114: "f32[112]"; primals_115: "f32[112]"; primals_116: "f32[672, 112, 1, 1]"; primals_117: "f32[672]"; primals_118: "f32[672]"; primals_119: "f32[672, 1, 3, 3]"; primals_120: "f32[672]"; primals_121: "f32[672]"; primals_122: "f32[168, 672, 1, 1]"; primals_123: "f32[168]"; primals_124: "f32[672, 168, 1, 1]"; primals_125: "f32[672]"; primals_126: "f32[112, 672, 1, 1]"; primals_127: "f32[112]"; primals_128: "f32[112]"; primals_129: "f32[672, 112, 1, 1]"; primals_130: "f32[672]"; primals_131: "f32[672]"; primals_132: "f32[672, 1, 5, 5]"; primals_133: "f32[672]"; primals_134: "f32[672]"; primals_135: "f32[168, 672, 1, 1]"; primals_136: "f32[168]"; primals_137: "f32[672, 168, 1, 1]"; primals_138: "f32[672]"; primals_139: "f32[160, 672, 1, 1]"; primals_140: "f32[160]"; primals_141: "f32[160]"; primals_142: "f32[960, 160, 1, 1]"; primals_143: "f32[960]"; primals_144: "f32[960]"; primals_145: "f32[960, 1, 5, 5]"; primals_146: "f32[960]"; primals_147: "f32[960]"; primals_148: "f32[240, 960, 1, 1]"; primals_149: "f32[240]"; primals_150: "f32[960, 240, 1, 1]"; primals_151: "f32[960]"; primals_152: "f32[160, 960, 1, 1]"; primals_153: "f32[160]"; primals_154: "f32[160]"; primals_155: "f32[960, 160, 1, 1]"; primals_156: "f32[960]"; primals_157: "f32[960]"; primals_158: "f32[960, 1, 5, 5]"; primals_159: "f32[960]"; primals_160: "f32[960]"; primals_161: "f32[240, 960, 1, 1]"; primals_162: "f32[240]"; primals_163: "f32[960, 240, 1, 1]"; primals_164: "f32[960]"; primals_165: "f32[160, 960, 1, 1]"; primals_166: "f32[160]"; primals_167: "f32[160]"; primals_168: "f32[960, 160, 1, 1]"; primals_169: "f32[960]"; primals_170: "f32[960]"; primals_171: "f32[1280, 960]"; primals_172: "f32[1280]"; primals_173: "f32[1000, 1280]"; primals_174: "f32[1000]"; primals_175: "f32[16]"; primals_176: "f32[16]"; primals_177: "i64[]"; primals_178: "f32[16]"; primals_179: "f32[16]"; primals_180: "i64[]"; primals_181: "f32[16]"; primals_182: "f32[16]"; primals_183: "i64[]"; primals_184: "f32[64]"; primals_185: "f32[64]"; primals_186: "i64[]"; primals_187: "f32[64]"; primals_188: "f32[64]"; primals_189: "i64[]"; primals_190: "f32[24]"; primals_191: "f32[24]"; primals_192: "i64[]"; primals_193: "f32[72]"; primals_194: "f32[72]"; primals_195: "i64[]"; primals_196: "f32[72]"; primals_197: "f32[72]"; primals_198: "i64[]"; primals_199: "f32[24]"; primals_200: "f32[24]"; primals_201: "i64[]"; primals_202: "f32[72]"; primals_203: "f32[72]"; primals_204: "i64[]"; primals_205: "f32[72]"; primals_206: "f32[72]"; primals_207: "i64[]"; primals_208: "f32[40]"; primals_209: "f32[40]"; primals_210: "i64[]"; primals_211: "f32[120]"; primals_212: "f32[120]"; primals_213: "i64[]"; primals_214: "f32[120]"; primals_215: "f32[120]"; primals_216: "i64[]"; primals_217: "f32[40]"; primals_218: "f32[40]"; primals_219: "i64[]"; primals_220: "f32[120]"; primals_221: "f32[120]"; primals_222: "i64[]"; primals_223: "f32[120]"; primals_224: "f32[120]"; primals_225: "i64[]"; primals_226: "f32[40]"; primals_227: "f32[40]"; primals_228: "i64[]"; primals_229: "f32[240]"; primals_230: "f32[240]"; primals_231: "i64[]"; primals_232: "f32[240]"; primals_233: "f32[240]"; primals_234: "i64[]"; primals_235: "f32[80]"; primals_236: "f32[80]"; primals_237: "i64[]"; primals_238: "f32[200]"; primals_239: "f32[200]"; primals_240: "i64[]"; primals_241: "f32[200]"; primals_242: "f32[200]"; primals_243: "i64[]"; primals_244: "f32[80]"; primals_245: "f32[80]"; primals_246: "i64[]"; primals_247: "f32[184]"; primals_248: "f32[184]"; primals_249: "i64[]"; primals_250: "f32[184]"; primals_251: "f32[184]"; primals_252: "i64[]"; primals_253: "f32[80]"; primals_254: "f32[80]"; primals_255: "i64[]"; primals_256: "f32[184]"; primals_257: "f32[184]"; primals_258: "i64[]"; primals_259: "f32[184]"; primals_260: "f32[184]"; primals_261: "i64[]"; primals_262: "f32[80]"; primals_263: "f32[80]"; primals_264: "i64[]"; primals_265: "f32[480]"; primals_266: "f32[480]"; primals_267: "i64[]"; primals_268: "f32[480]"; primals_269: "f32[480]"; primals_270: "i64[]"; primals_271: "f32[112]"; primals_272: "f32[112]"; primals_273: "i64[]"; primals_274: "f32[672]"; primals_275: "f32[672]"; primals_276: "i64[]"; primals_277: "f32[672]"; primals_278: "f32[672]"; primals_279: "i64[]"; primals_280: "f32[112]"; primals_281: "f32[112]"; primals_282: "i64[]"; primals_283: "f32[672]"; primals_284: "f32[672]"; primals_285: "i64[]"; primals_286: "f32[672]"; primals_287: "f32[672]"; primals_288: "i64[]"; primals_289: "f32[160]"; primals_290: "f32[160]"; primals_291: "i64[]"; primals_292: "f32[960]"; primals_293: "f32[960]"; primals_294: "i64[]"; primals_295: "f32[960]"; primals_296: "f32[960]"; primals_297: "i64[]"; primals_298: "f32[160]"; primals_299: "f32[160]"; primals_300: "i64[]"; primals_301: "f32[960]"; primals_302: "f32[960]"; primals_303: "i64[]"; primals_304: "f32[960]"; primals_305: "f32[960]"; primals_306: "i64[]"; primals_307: "f32[160]"; primals_308: "f32[160]"; primals_309: "i64[]"; primals_310: "f32[960]"; primals_311: "f32[960]"; primals_312: "i64[]"; primals_313: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    convolution: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_313, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    convert_element_type_1: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_176, torch.float32)
    add: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    clone: "f32[4, 16, 112, 112]" = torch.ops.aten.clone.default(add_1)
    add_2: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_1, 3)
    clamp_min: "f32[4, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max: "f32[4, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_3: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, clamp_max);  add_1 = clamp_max = None
    div: "f32[4, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_1: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(div, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    convert_element_type_2: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    convert_element_type_3: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_179, torch.float32)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_3, 0.001);  convert_element_type_3 = None
    sqrt_1: "f32[16]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_5: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    relu: "f32[4, 16, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    convolution_2: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    convert_element_type_5: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_182, torch.float32)
    add_5: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_5, 0.001);  convert_element_type_5 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_7: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_8: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_9: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_21);  mul_8 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_6: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_23);  mul_9 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_7: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_6, div);  add_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_3: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(add_7, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_185, torch.float32)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 0.001);  convert_element_type_7 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_11: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_12: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_9: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
    relu_1: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_4: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)
    convert_element_type_8: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    convert_element_type_9: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_188, torch.float32)
    add_10: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_9, 0.001);  convert_element_type_9 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_14: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_15: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_11: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
    relu_2: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    convolution_5: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_10: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    convert_element_type_11: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_191, torch.float32)
    add_12: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_11, 0.001);  convert_element_type_11 = None
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_16: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_17: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_18: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_45);  mul_17 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_13: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_47);  mul_18 = unsqueeze_47 = None
    convolution_6: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_13, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_12: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    convert_element_type_13: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_194, torch.float32)
    add_14: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_13, 0.001);  convert_element_type_13 = None
    sqrt_6: "f32[72]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_6: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_19: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_51: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_20: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_21: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
    unsqueeze_54: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_15: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
    relu_3: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    convolution_7: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    convert_element_type_14: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    convert_element_type_15: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_197, torch.float32)
    add_16: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_15, 0.001);  convert_element_type_15 = None
    sqrt_7: "f32[72]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_7: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_22: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
    unsqueeze_59: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_23: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_24: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_61);  mul_23 = unsqueeze_61 = None
    unsqueeze_62: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_17: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_63);  mul_24 = unsqueeze_63 = None
    relu_4: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    convolution_8: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_16: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_199, torch.float32)
    convert_element_type_17: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_200, torch.float32)
    add_18: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_17, 0.001);  convert_element_type_17 = None
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_25: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_25, -1);  mul_25 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_26: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_27: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_69);  mul_26 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_71);  mul_27 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_20: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_13);  add_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_9: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_20, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_18: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_202, torch.float32)
    convert_element_type_19: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_203, torch.float32)
    add_21: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_19, 0.001);  convert_element_type_19 = None
    sqrt_9: "f32[72]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_9: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_28: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_75: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_29: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_30: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_77);  mul_29 = unsqueeze_77 = None
    unsqueeze_78: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_22: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_79);  mul_30 = unsqueeze_79 = None
    relu_5: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    convolution_10: "f32[4, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_31, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    convert_element_type_20: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_205, torch.float32)
    convert_element_type_21: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_206, torch.float32)
    add_23: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_21, 0.001);  convert_element_type_21 = None
    sqrt_10: "f32[72]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_10: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_31: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
    unsqueeze_83: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_32: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_33: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
    unsqueeze_86: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_24: "f32[4, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
    relu_6: "f32[4, 72, 28, 28]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean: "f32[4, 72, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_11: "f32[4, 24, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_34, primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_7: "f32[4, 24, 1, 1]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    alias_7: "f32[4, 24, 1, 1]" = torch.ops.aten.alias.default(relu_7)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_12: "f32[4, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_36, primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_25: "f32[4, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_12, 3)
    clamp_min_1: "f32[4, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_25, 0);  add_25 = None
    clamp_max_1: "f32[4, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[4, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_34: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(div_1, relu_6)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_13: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_34, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_22: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_208, torch.float32)
    convert_element_type_23: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_209, torch.float32)
    add_26: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_23, 0.001);  convert_element_type_23 = None
    sqrt_11: "f32[40]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_11: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_35: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  unsqueeze_89 = None
    mul_36: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_93: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_37: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_93);  mul_36 = unsqueeze_93 = None
    unsqueeze_94: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_95: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_27: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_95);  mul_37 = unsqueeze_95 = None
    convolution_14: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_27, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_211, torch.float32)
    convert_element_type_25: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_212, torch.float32)
    add_28: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_25, 0.001);  convert_element_type_25 = None
    sqrt_12: "f32[120]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_38: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
    unsqueeze_99: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_97);  unsqueeze_97 = None
    mul_39: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
    unsqueeze_101: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_40: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_101);  mul_39 = unsqueeze_101 = None
    unsqueeze_102: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
    unsqueeze_103: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_103);  mul_40 = unsqueeze_103 = None
    relu_8: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    convolution_15: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_44, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    convert_element_type_26: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    convert_element_type_27: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_215, torch.float32)
    add_30: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_27, 0.001);  convert_element_type_27 = None
    sqrt_13: "f32[120]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_13: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_41: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_107: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_105);  unsqueeze_105 = None
    mul_42: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_109: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_43: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_109);  mul_42 = unsqueeze_109 = None
    unsqueeze_110: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_111: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_31: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_111);  mul_43 = unsqueeze_111 = None
    relu_9: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_1: "f32[4, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_9, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_16: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_47, primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_10: "f32[4, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    alias_10: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(relu_10)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_17: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_49, primals_50, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_32: "f32[4, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_17, 3)
    clamp_min_2: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_32, 0);  add_32 = None
    clamp_max_2: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[4, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_44: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_2, relu_9)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_18: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_44, primals_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    convert_element_type_29: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_218, torch.float32)
    add_33: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_29, 0.001);  convert_element_type_29 = None
    sqrt_14: "f32[40]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_45: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_113);  unsqueeze_113 = None
    mul_46: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1)
    unsqueeze_117: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_47: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_117);  mul_46 = unsqueeze_117 = None
    unsqueeze_118: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1);  primals_53 = None
    unsqueeze_119: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_119);  mul_47 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_35: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_34, add_27);  add_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_19: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_35, primals_54, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_30: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    convert_element_type_31: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_221, torch.float32)
    add_36: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_31, 0.001);  convert_element_type_31 = None
    sqrt_15: "f32[120]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_15: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_48: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_123: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_121);  unsqueeze_121 = None
    mul_49: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_125: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_50: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_125);  mul_49 = unsqueeze_125 = None
    unsqueeze_126: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_127: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_37: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_127);  mul_50 = unsqueeze_127 = None
    relu_11: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    convolution_20: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_57, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    convert_element_type_32: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    convert_element_type_33: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_224, torch.float32)
    add_38: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_33, 0.001);  convert_element_type_33 = None
    sqrt_16: "f32[120]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_51: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_131: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_129);  unsqueeze_129 = None
    mul_52: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_133: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_53: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_133);  mul_52 = unsqueeze_133 = None
    unsqueeze_134: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_135: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_135);  mul_53 = unsqueeze_135 = None
    relu_12: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_2: "f32[4, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_12, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_21: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_60, primals_61, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_13: "f32[4, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    alias_13: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(relu_13)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_22: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_62, primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_40: "f32[4, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3)
    clamp_min_3: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_40, 0);  add_40 = None
    clamp_max_3: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[4, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_54: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_3, relu_12)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_23: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_54, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    convert_element_type_35: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_227, torch.float32)
    add_41: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_35, 0.001);  convert_element_type_35 = None
    sqrt_17: "f32[40]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_17: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_55: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_55, -1);  mul_55 = None
    unsqueeze_139: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_137);  unsqueeze_137 = None
    mul_56: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_141: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_57: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_141);  mul_56 = unsqueeze_141 = None
    unsqueeze_142: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_143: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_42: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_143);  mul_57 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_43: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_42, add_35);  add_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_24: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_43, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_36: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    convert_element_type_37: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_230, torch.float32)
    add_44: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_37, 0.001);  convert_element_type_37 = None
    sqrt_18: "f32[240]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_18: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_58: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_58, -1);  mul_58 = None
    unsqueeze_147: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_145);  unsqueeze_145 = None
    mul_59: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_149: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_60: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_149);  mul_59 = unsqueeze_149 = None
    unsqueeze_150: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_151: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_45: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_151);  mul_60 = unsqueeze_151 = None
    clone_1: "f32[4, 240, 28, 28]" = torch.ops.aten.clone.default(add_45)
    add_46: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(add_45, 3)
    clamp_min_4: "f32[4, 240, 28, 28]" = torch.ops.aten.clamp_min.default(add_46, 0);  add_46 = None
    clamp_max_4: "f32[4, 240, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_61: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_45, clamp_max_4);  add_45 = clamp_max_4 = None
    div_4: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Tensor(mul_61, 6);  mul_61 = None
    convolution_25: "f32[4, 240, 14, 14]" = torch.ops.aten.convolution.default(div_4, primals_70, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    convert_element_type_38: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    convert_element_type_39: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_233, torch.float32)
    add_47: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_39, 0.001);  convert_element_type_39 = None
    sqrt_19: "f32[240]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_19: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_62: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_155: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_153);  unsqueeze_153 = None
    mul_63: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_157: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_64: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_157);  mul_63 = unsqueeze_157 = None
    unsqueeze_158: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_159: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_48: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_159);  mul_64 = unsqueeze_159 = None
    clone_2: "f32[4, 240, 14, 14]" = torch.ops.aten.clone.default(add_48)
    add_49: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_48, 3)
    clamp_min_5: "f32[4, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_49, 0);  add_49 = None
    clamp_max_5: "f32[4, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_65: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, clamp_max_5);  add_48 = clamp_max_5 = None
    div_5: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_65, 6);  mul_65 = None
    convolution_26: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_5, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_40: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    convert_element_type_41: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_236, torch.float32)
    add_50: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_41, 0.001);  convert_element_type_41 = None
    sqrt_20: "f32[80]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_20: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_66: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_163: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_161);  unsqueeze_161 = None
    mul_67: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_165: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_68: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_165);  mul_67 = unsqueeze_165 = None
    unsqueeze_166: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_167: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_51: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_167);  mul_68 = unsqueeze_167 = None
    convolution_27: "f32[4, 200, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_42: "f32[200]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    convert_element_type_43: "f32[200]" = torch.ops.prims.convert_element_type.default(primals_239, torch.float32)
    add_52: "f32[200]" = torch.ops.aten.add.Tensor(convert_element_type_43, 0.001);  convert_element_type_43 = None
    sqrt_21: "f32[200]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_21: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_69: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_171: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_169);  unsqueeze_169 = None
    mul_70: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_173: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_71: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_173);  mul_70 = unsqueeze_173 = None
    unsqueeze_174: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_175: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_53: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_175);  mul_71 = unsqueeze_175 = None
    clone_3: "f32[4, 200, 14, 14]" = torch.ops.aten.clone.default(add_53)
    add_54: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_53, 3)
    clamp_min_6: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_54, 0);  add_54 = None
    clamp_max_6: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_72: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_53, clamp_max_6);  add_53 = clamp_max_6 = None
    div_6: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_72, 6);  mul_72 = None
    convolution_28: "f32[4, 200, 14, 14]" = torch.ops.aten.convolution.default(div_6, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200)
    convert_element_type_44: "f32[200]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    convert_element_type_45: "f32[200]" = torch.ops.prims.convert_element_type.default(primals_242, torch.float32)
    add_55: "f32[200]" = torch.ops.aten.add.Tensor(convert_element_type_45, 0.001);  convert_element_type_45 = None
    sqrt_22: "f32[200]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_22: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_73: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
    unsqueeze_179: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_177);  unsqueeze_177 = None
    mul_74: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_181: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_75: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_181);  mul_74 = unsqueeze_181 = None
    unsqueeze_182: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_183: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_56: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_183);  mul_75 = unsqueeze_183 = None
    clone_4: "f32[4, 200, 14, 14]" = torch.ops.aten.clone.default(add_56)
    add_57: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_56, 3)
    clamp_min_7: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_57, 0);  add_57 = None
    clamp_max_7: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_76: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_56, clamp_max_7);  add_56 = clamp_max_7 = None
    div_7: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_76, 6);  mul_76 = None
    convolution_29: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_7, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_46: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    convert_element_type_47: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_245, torch.float32)
    add_58: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_47, 0.001);  convert_element_type_47 = None
    sqrt_23: "f32[80]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_23: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_77: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_77, -1);  mul_77 = None
    unsqueeze_187: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_185);  unsqueeze_185 = None
    mul_78: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_189: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_79: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_189);  mul_78 = unsqueeze_189 = None
    unsqueeze_190: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_191: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_59: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_191);  mul_79 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_60: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_59, add_51);  add_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_30: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(add_60, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_48: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    convert_element_type_49: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_248, torch.float32)
    add_61: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_49, 0.001);  convert_element_type_49 = None
    sqrt_24: "f32[184]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_24: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_80: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
    unsqueeze_195: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_193);  unsqueeze_193 = None
    mul_81: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_197: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_82: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_197);  mul_81 = unsqueeze_197 = None
    unsqueeze_198: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_199: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_62: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_199);  mul_82 = unsqueeze_199 = None
    clone_5: "f32[4, 184, 14, 14]" = torch.ops.aten.clone.default(add_62)
    add_63: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_62, 3)
    clamp_min_8: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_63, 0);  add_63 = None
    clamp_max_8: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_83: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_62, clamp_max_8);  add_62 = clamp_max_8 = None
    div_8: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_83, 6);  mul_83 = None
    convolution_31: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(div_8, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    convert_element_type_50: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    convert_element_type_51: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_251, torch.float32)
    add_64: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_51, 0.001);  convert_element_type_51 = None
    sqrt_25: "f32[184]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_25: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_84: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_203: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_201);  unsqueeze_201 = None
    mul_85: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_205: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_86: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_205);  mul_85 = unsqueeze_205 = None
    unsqueeze_206: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_207: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_65: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_207);  mul_86 = unsqueeze_207 = None
    clone_6: "f32[4, 184, 14, 14]" = torch.ops.aten.clone.default(add_65)
    add_66: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_65, 3)
    clamp_min_9: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_9: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_87: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_65, clamp_max_9);  add_65 = clamp_max_9 = None
    div_9: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_87, 6);  mul_87 = None
    convolution_32: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_9, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_52: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    convert_element_type_53: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_254, torch.float32)
    add_67: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_53, 0.001);  convert_element_type_53 = None
    sqrt_26: "f32[80]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_26: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_88: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_211: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_209);  unsqueeze_209 = None
    mul_89: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_213: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_90: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_213);  mul_89 = unsqueeze_213 = None
    unsqueeze_214: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_215: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_68: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_215);  mul_90 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_69: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_68, add_60);  add_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_33: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(add_69, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_54: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    convert_element_type_55: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_257, torch.float32)
    add_70: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_55, 0.001);  convert_element_type_55 = None
    sqrt_27: "f32[184]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_27: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_91: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_219: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_217);  unsqueeze_217 = None
    mul_92: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_221: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_93: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_221);  mul_92 = unsqueeze_221 = None
    unsqueeze_222: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_223: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_71: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_223);  mul_93 = unsqueeze_223 = None
    clone_7: "f32[4, 184, 14, 14]" = torch.ops.aten.clone.default(add_71)
    add_72: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_71, 3)
    clamp_min_10: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_72, 0);  add_72 = None
    clamp_max_10: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_94: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_71, clamp_max_10);  add_71 = clamp_max_10 = None
    div_10: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_94, 6);  mul_94 = None
    convolution_34: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(div_10, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    convert_element_type_56: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    convert_element_type_57: "f32[184]" = torch.ops.prims.convert_element_type.default(primals_260, torch.float32)
    add_73: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_57, 0.001);  convert_element_type_57 = None
    sqrt_28: "f32[184]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_28: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_95: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_227: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_225);  unsqueeze_225 = None
    mul_96: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_229: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_97: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_229);  mul_96 = unsqueeze_229 = None
    unsqueeze_230: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_231: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_74: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_231);  mul_97 = unsqueeze_231 = None
    clone_8: "f32[4, 184, 14, 14]" = torch.ops.aten.clone.default(add_74)
    add_75: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_74, 3)
    clamp_min_11: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_75, 0);  add_75 = None
    clamp_max_11: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_98: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_74, clamp_max_11);  add_74 = clamp_max_11 = None
    div_11: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_98, 6);  mul_98 = None
    convolution_35: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_11, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_58: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    convert_element_type_59: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_263, torch.float32)
    add_76: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_59, 0.001);  convert_element_type_59 = None
    sqrt_29: "f32[80]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_29: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_99: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_235: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_233);  unsqueeze_233 = None
    mul_100: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_237: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_101: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_237);  mul_100 = unsqueeze_237 = None
    unsqueeze_238: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_239: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_77: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_239);  mul_101 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_78: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_77, add_69);  add_77 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_36: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_78, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_60: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    convert_element_type_61: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_266, torch.float32)
    add_79: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_61, 0.001);  convert_element_type_61 = None
    sqrt_30: "f32[480]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_30: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_102: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_243: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_241);  unsqueeze_241 = None
    mul_103: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_245: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_104: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_245);  mul_103 = unsqueeze_245 = None
    unsqueeze_246: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_247: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_80: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_247);  mul_104 = unsqueeze_247 = None
    clone_9: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_80)
    add_81: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_80, 3)
    clamp_min_12: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_81, 0);  add_81 = None
    clamp_max_12: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_105: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_80, clamp_max_12);  add_80 = clamp_max_12 = None
    div_12: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_105, 6);  mul_105 = None
    convolution_37: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(div_12, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    convert_element_type_62: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    convert_element_type_63: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_269, torch.float32)
    add_82: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_63, 0.001);  convert_element_type_63 = None
    sqrt_31: "f32[480]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_31: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_106: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
    unsqueeze_251: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_249);  unsqueeze_249 = None
    mul_107: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_253: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_108: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_253);  mul_107 = unsqueeze_253 = None
    unsqueeze_254: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_255: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_83: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_255);  mul_108 = unsqueeze_255 = None
    clone_10: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_83)
    add_84: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_83, 3)
    clamp_min_13: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_84, 0);  add_84 = None
    clamp_max_13: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_109: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_83, clamp_max_13);  add_83 = clamp_max_13 = None
    div_13: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_109, 6);  mul_109 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_3: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(div_13, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_38: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_109, primals_110, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_14: "f32[4, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
    alias_14: "f32[4, 120, 1, 1]" = torch.ops.aten.alias.default(relu_14)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_39: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_14, primals_111, primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_112 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_85: "f32[4, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
    clamp_min_14: "f32[4, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_85, 0);  add_85 = None
    clamp_max_14: "f32[4, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    div_14: "f32[4, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_110: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(div_14, div_13)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_40: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_110, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_64: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    convert_element_type_65: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_272, torch.float32)
    add_86: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_65, 0.001);  convert_element_type_65 = None
    sqrt_32: "f32[112]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_32: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_111: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_259: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  unsqueeze_257 = None
    mul_112: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_261: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_113: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_261);  mul_112 = unsqueeze_261 = None
    unsqueeze_262: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_263: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_87: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_263);  mul_113 = unsqueeze_263 = None
    convolution_41: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_87, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_66: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    convert_element_type_67: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_275, torch.float32)
    add_88: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_67, 0.001);  convert_element_type_67 = None
    sqrt_33: "f32[672]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_33: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_114: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_267: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  unsqueeze_265 = None
    mul_115: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_269: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_116: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_269);  mul_115 = unsqueeze_269 = None
    unsqueeze_270: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_271: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_89: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_271);  mul_116 = unsqueeze_271 = None
    clone_11: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_89)
    add_90: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_89, 3)
    clamp_min_15: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_90, 0);  add_90 = None
    clamp_max_15: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_117: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_89, clamp_max_15);  add_89 = clamp_max_15 = None
    div_15: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_117, 6);  mul_117 = None
    convolution_42: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(div_15, primals_119, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672)
    convert_element_type_68: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    convert_element_type_69: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_278, torch.float32)
    add_91: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_69, 0.001);  convert_element_type_69 = None
    sqrt_34: "f32[672]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_34: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_118: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_118, -1);  mul_118 = None
    unsqueeze_275: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  unsqueeze_273 = None
    mul_119: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
    unsqueeze_277: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_120: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_277);  mul_119 = unsqueeze_277 = None
    unsqueeze_278: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
    unsqueeze_279: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_92: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_279);  mul_120 = unsqueeze_279 = None
    clone_12: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_92)
    add_93: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_92, 3)
    clamp_min_16: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_93, 0);  add_93 = None
    clamp_max_16: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_121: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_92, clamp_max_16);  add_92 = clamp_max_16 = None
    div_16: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_121, 6);  mul_121 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_4: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(div_16, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_43: "f32[4, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_15: "f32[4, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
    alias_15: "f32[4, 168, 1, 1]" = torch.ops.aten.alias.default(relu_15)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_44: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_124, primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_94: "f32[4, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_44, 3)
    clamp_min_17: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_94, 0);  add_94 = None
    clamp_max_17: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    div_17: "f32[4, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_122: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(div_17, div_16)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_45: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_122, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_70: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    convert_element_type_71: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_281, torch.float32)
    add_95: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_71, 0.001);  convert_element_type_71 = None
    sqrt_35: "f32[112]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_35: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_123: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_283: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_281);  unsqueeze_281 = None
    mul_124: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_285: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_125: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_285);  mul_124 = unsqueeze_285 = None
    unsqueeze_286: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_287: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_96: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_287);  mul_125 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_97: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_96, add_87);  add_96 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_46: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_97, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_72: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    convert_element_type_73: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_284, torch.float32)
    add_98: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_73, 0.001);  convert_element_type_73 = None
    sqrt_36: "f32[672]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_36: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_126: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_291: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_289);  unsqueeze_289 = None
    mul_127: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_293: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_128: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_293);  mul_127 = unsqueeze_293 = None
    unsqueeze_294: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_295: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_99: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_295);  mul_128 = unsqueeze_295 = None
    clone_13: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_99)
    add_100: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_99, 3)
    clamp_min_18: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_100, 0);  add_100 = None
    clamp_max_18: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_129: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_99, clamp_max_18);  add_99 = clamp_max_18 = None
    div_18: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_129, 6);  mul_129 = None
    convolution_47: "f32[4, 672, 7, 7]" = torch.ops.aten.convolution.default(div_18, primals_132, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    convert_element_type_74: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    convert_element_type_75: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_287, torch.float32)
    add_101: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_75, 0.001);  convert_element_type_75 = None
    sqrt_37: "f32[672]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_37: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_130: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
    unsqueeze_299: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_297);  unsqueeze_297 = None
    mul_131: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_301: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_132: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_301);  mul_131 = unsqueeze_301 = None
    unsqueeze_302: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_303: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_102: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_303);  mul_132 = unsqueeze_303 = None
    clone_14: "f32[4, 672, 7, 7]" = torch.ops.aten.clone.default(add_102)
    add_103: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(add_102, 3)
    clamp_min_19: "f32[4, 672, 7, 7]" = torch.ops.aten.clamp_min.default(add_103, 0);  add_103 = None
    clamp_max_19: "f32[4, 672, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_133: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, clamp_max_19);  add_102 = clamp_max_19 = None
    div_19: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Tensor(mul_133, 6);  mul_133 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_5: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(div_19, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_48: "f32[4, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_135, primals_136, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_136 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_16: "f32[4, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
    alias_16: "f32[4, 168, 1, 1]" = torch.ops.aten.alias.default(relu_16)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_49: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_16, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_104: "f32[4, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_49, 3)
    clamp_min_20: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
    clamp_max_20: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    div_20: "f32[4, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_134: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(div_20, div_19)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_50: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_134, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_76: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    convert_element_type_77: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_290, torch.float32)
    add_105: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_77, 0.001);  convert_element_type_77 = None
    sqrt_38: "f32[160]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_38: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_135: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_307: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  unsqueeze_305 = None
    mul_136: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_309: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_137: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_309);  mul_136 = unsqueeze_309 = None
    unsqueeze_310: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_311: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_106: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_311);  mul_137 = unsqueeze_311 = None
    convolution_51: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_106, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_78: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    convert_element_type_79: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_293, torch.float32)
    add_107: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_79, 0.001);  convert_element_type_79 = None
    sqrt_39: "f32[960]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_39: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_138: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_315: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  unsqueeze_313 = None
    mul_139: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_317: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_140: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_317);  mul_139 = unsqueeze_317 = None
    unsqueeze_318: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_319: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_108: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_319);  mul_140 = unsqueeze_319 = None
    clone_15: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_108)
    add_109: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_108, 3)
    clamp_min_21: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
    clamp_max_21: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_141: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_108, clamp_max_21);  add_108 = clamp_max_21 = None
    div_21: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_141, 6);  mul_141 = None
    convolution_52: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(div_21, primals_145, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    convert_element_type_80: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    convert_element_type_81: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_296, torch.float32)
    add_110: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_81, 0.001);  convert_element_type_81 = None
    sqrt_40: "f32[960]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_40: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_142: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
    unsqueeze_323: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  unsqueeze_321 = None
    mul_143: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_325: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_144: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_325);  mul_143 = unsqueeze_325 = None
    unsqueeze_326: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_327: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_111: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_327);  mul_144 = unsqueeze_327 = None
    clone_16: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_111)
    add_112: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_111, 3)
    clamp_min_22: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_112, 0);  add_112 = None
    clamp_max_22: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_145: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_111, clamp_max_22);  add_111 = clamp_max_22 = None
    div_22: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_6: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_22, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_53: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_148, primals_149, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_149 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_17: "f32[4, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    alias_17: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(relu_17)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_54: "f32[4, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_17, primals_150, primals_151, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_113: "f32[4, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_54, 3)
    clamp_min_23: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
    clamp_max_23: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    div_23: "f32[4, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_146: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_23, div_22)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_55: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_146, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_82: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    convert_element_type_83: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_299, torch.float32)
    add_114: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_83, 0.001);  convert_element_type_83 = None
    sqrt_41: "f32[160]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_41: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_147: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_331: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_329);  unsqueeze_329 = None
    mul_148: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1)
    unsqueeze_333: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_149: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_333);  mul_148 = unsqueeze_333 = None
    unsqueeze_334: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
    unsqueeze_335: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_115: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_335);  mul_149 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_116: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_115, add_106);  add_115 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_56: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_116, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_84: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    convert_element_type_85: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_302, torch.float32)
    add_117: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_85, 0.001);  convert_element_type_85 = None
    sqrt_42: "f32[960]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_42: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_150: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_339: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_337);  unsqueeze_337 = None
    mul_151: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
    unsqueeze_341: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_152: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_341);  mul_151 = unsqueeze_341 = None
    unsqueeze_342: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
    unsqueeze_343: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_118: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_343);  mul_152 = unsqueeze_343 = None
    clone_17: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_118)
    add_119: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_118, 3)
    clamp_min_24: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
    clamp_max_24: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_153: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_118, clamp_max_24);  add_118 = clamp_max_24 = None
    div_24: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_153, 6);  mul_153 = None
    convolution_57: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(div_24, primals_158, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    convert_element_type_86: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    convert_element_type_87: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_305, torch.float32)
    add_120: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_87, 0.001);  convert_element_type_87 = None
    sqrt_43: "f32[960]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_43: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_154: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_154, -1);  mul_154 = None
    unsqueeze_347: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  unsqueeze_345 = None
    mul_155: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
    unsqueeze_349: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_156: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_349);  mul_155 = unsqueeze_349 = None
    unsqueeze_350: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
    unsqueeze_351: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_121: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_351);  mul_156 = unsqueeze_351 = None
    clone_18: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_121)
    add_122: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_121, 3)
    clamp_min_25: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_122, 0);  add_122 = None
    clamp_max_25: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_157: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, clamp_max_25);  add_121 = clamp_max_25 = None
    div_25: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_157, 6);  mul_157 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_7: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_25, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_58: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_18: "f32[4, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
    alias_18: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(relu_18)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_59: "f32[4, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_18, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_123: "f32[4, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3)
    clamp_min_26: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_123, 0);  add_123 = None
    clamp_max_26: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    div_26: "f32[4, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_158: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_26, div_25)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_60: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_158, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_88: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    convert_element_type_89: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_308, torch.float32)
    add_124: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_89, 0.001);  convert_element_type_89 = None
    sqrt_44: "f32[160]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_44: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_159: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_355: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_353);  unsqueeze_353 = None
    mul_160: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
    unsqueeze_357: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_161: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_357);  mul_160 = unsqueeze_357 = None
    unsqueeze_358: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
    unsqueeze_359: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_125: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_359);  mul_161 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_126: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_125, add_116);  add_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    convolution_61: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_126, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_90: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    convert_element_type_91: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_311, torch.float32)
    add_127: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_91, 0.001);  convert_element_type_91 = None
    sqrt_45: "f32[960]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_45: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_162: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_363: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_361);  unsqueeze_361 = None
    mul_163: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1)
    unsqueeze_365: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_164: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_365);  mul_163 = unsqueeze_365 = None
    unsqueeze_366: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
    unsqueeze_367: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_128: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_367);  mul_164 = unsqueeze_367 = None
    clone_19: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_128)
    add_129: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_128, 3)
    clamp_min_27: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_129, 0);  add_129 = None
    clamp_max_27: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_165: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_128, clamp_max_27);  add_128 = clamp_max_27 = None
    div_27: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_165, 6);  mul_165 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
    mean_8: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_27, [-1, -2], True);  div_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
    view: "f32[4, 960]" = torch.ops.aten.view.default(mean_8, [4, 960]);  mean_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
    permute: "f32[960, 1280]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_172, view, permute);  primals_172 = None
    clone_20: "f32[4, 1280]" = torch.ops.aten.clone.default(addmm)
    add_130: "f32[4, 1280]" = torch.ops.aten.add.Tensor(addmm, 3)
    clamp_min_28: "f32[4, 1280]" = torch.ops.aten.clamp_min.default(add_130, 0);  add_130 = None
    clamp_max_28: "f32[4, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_166: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(addmm, clamp_max_28);  addmm = clamp_max_28 = None
    div_28: "f32[4, 1280]" = torch.ops.aten.div.Tensor(mul_166, 6);  mul_166 = None
    permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_1: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_174, div_28, permute_1);  primals_174 = None
    permute_2: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
    permute_3: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_3, div_28);  permute_3 = div_28 = None
    permute_4: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_5: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    lt: "b8[4, 1280]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le: "b8[4, 1280]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_29: "f32[4, 1280]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_131: "f32[4, 1280]" = torch.ops.aten.add.Tensor(div_29, 0.5);  div_29 = None
    mul_167: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(mm, add_131);  add_131 = None
    where: "f32[4, 1280]" = torch.ops.aten.where.self(le, mul_167, mm);  le = mul_167 = mm = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 1280]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = scalar_tensor = where = None
    permute_6: "f32[1280, 960]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_2: "f32[4, 960]" = torch.ops.aten.mm.default(where_1, permute_6);  permute_6 = None
    permute_7: "f32[1280, 4]" = torch.ops.aten.permute.default(where_1, [1, 0])
    mm_3: "f32[1280, 960]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
    permute_8: "f32[960, 1280]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
    view_2: "f32[1280]" = torch.ops.aten.view.default(sum_2, [1280]);  sum_2 = None
    permute_9: "f32[1280, 960]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
    view_3: "f32[4, 960, 1, 1]" = torch.ops.aten.view.default(mm_2, [4, 960, 1, 1]);  mm_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
    expand: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(view_3, [4, 960, 7, 7]);  view_3 = None
    div_30: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    lt_1: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_1: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_31: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_132: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_31, 0.5);  div_31 = None
    mul_168: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_30, add_132);  add_132 = None
    where_2: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_1, mul_168, div_30);  le_1 = mul_168 = div_30 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_1, scalar_tensor_1, where_2);  lt_1 = scalar_tensor_1 = where_2 = None
    add_133: "f32[960]" = torch.ops.aten.add.Tensor(primals_311, 0.001);  primals_311 = None
    rsqrt: "f32[960]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_368: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_310, 0);  primals_310 = None
    unsqueeze_369: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_3: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_46: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_370);  convolution_61 = unsqueeze_370 = None
    mul_169: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_46);  sub_46 = None
    sum_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 2, 3]);  mul_169 = None
    mul_174: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt, primals_169);  primals_169 = None
    unsqueeze_377: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_174, 0);  mul_174 = None
    unsqueeze_378: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_175: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_379);  where_3 = unsqueeze_379 = None
    mul_176: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, rsqrt);  sum_4 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_175, add_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = add_126 = primals_168 = None
    getitem: "f32[4, 160, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[960, 160, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_134: "f32[160]" = torch.ops.aten.add.Tensor(primals_308, 0.001);  primals_308 = None
    rsqrt_1: "f32[160]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_380: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_307, 0);  primals_307 = None
    unsqueeze_381: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_47: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_382);  convolution_60 = unsqueeze_382 = None
    mul_177: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_47);  sub_47 = None
    sum_6: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3]);  mul_177 = None
    mul_182: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_166);  primals_166 = None
    unsqueeze_389: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
    unsqueeze_390: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_183: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_391);  unsqueeze_391 = None
    mul_184: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, rsqrt_1);  sum_6 = rsqrt_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_183, mul_158, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_183 = mul_158 = primals_165 = None
    getitem_3: "f32[4, 960, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[160, 960, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_185: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, div_26);  div_26 = None
    mul_186: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, div_25);  getitem_3 = div_25 = None
    sum_7: "f32[4, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2, 3], True);  mul_186 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt: "b8[4, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_59, -3.0)
    lt_2: "b8[4, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_59, 3.0);  convolution_59 = None
    bitwise_and: "b8[4, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    mul_187: "f32[4, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_7, 0.16666666666666666);  sum_7 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_187, scalar_tensor_2);  bitwise_and = mul_187 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, relu_18, primals_163, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = relu_18 = primals_163 = None
    getitem_6: "f32[4, 240, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[960, 240, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_19: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    le_2: "b8[4, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_19, 0);  alias_19 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 240, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_3, getitem_6);  le_2 = scalar_tensor_3 = getitem_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_9: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_7, primals_161, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_7 = primals_161 = None
    getitem_9: "f32[4, 960, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[240, 960, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_1: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 960, 7, 7]);  getitem_9 = None
    div_32: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_135: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_185, div_32);  mul_185 = div_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_3: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_3: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_33: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_136: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_33, 0.5);  div_33 = None
    mul_188: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_135, add_136);  add_136 = None
    where_6: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_3, mul_188, add_135);  le_3 = mul_188 = add_135 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_3, scalar_tensor_4, where_6);  lt_3 = scalar_tensor_4 = where_6 = None
    add_137: "f32[960]" = torch.ops.aten.add.Tensor(primals_305, 0.001);  primals_305 = None
    rsqrt_2: "f32[960]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_392: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_304, 0);  primals_304 = None
    unsqueeze_393: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_10: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_48: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_394);  convolution_57 = unsqueeze_394 = None
    mul_189: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_48);  sub_48 = None
    sum_11: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3]);  mul_189 = None
    mul_194: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_159);  primals_159 = None
    unsqueeze_401: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_194, 0);  mul_194 = None
    unsqueeze_402: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_195: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_403);  where_7 = unsqueeze_403 = None
    mul_196: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_2);  sum_11 = rsqrt_2 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_195, div_24, primals_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_195 = div_24 = primals_158 = None
    getitem_12: "f32[4, 960, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[960, 1, 5, 5]" = convolution_backward_4[1];  convolution_backward_4 = None
    lt_4: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_4: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_34: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_138: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
    mul_197: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_12, add_138);  add_138 = None
    where_8: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_4, mul_197, getitem_12);  le_4 = mul_197 = getitem_12 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_4, scalar_tensor_5, where_8);  lt_4 = scalar_tensor_5 = where_8 = None
    add_139: "f32[960]" = torch.ops.aten.add.Tensor(primals_302, 0.001);  primals_302 = None
    rsqrt_3: "f32[960]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_404: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_301, 0);  primals_301 = None
    unsqueeze_405: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_12: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_49: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_406);  convolution_56 = unsqueeze_406 = None
    mul_198: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_49);  sub_49 = None
    sum_13: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 2, 3]);  mul_198 = None
    mul_203: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_156);  primals_156 = None
    unsqueeze_413: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_203, 0);  mul_203 = None
    unsqueeze_414: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_204: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_415);  where_9 = unsqueeze_415 = None
    mul_205: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_3);  sum_13 = rsqrt_3 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_204, add_116, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_204 = add_116 = primals_155 = None
    getitem_15: "f32[4, 160, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[960, 160, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_140: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(getitem, getitem_15);  getitem = getitem_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_141: "f32[160]" = torch.ops.aten.add.Tensor(primals_299, 0.001);  primals_299 = None
    rsqrt_4: "f32[160]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_416: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_298, 0);  primals_298 = None
    unsqueeze_417: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_14: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 2, 3])
    sub_50: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_418);  convolution_55 = unsqueeze_418 = None
    mul_206: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_140, sub_50);  sub_50 = None
    sum_15: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 2, 3]);  mul_206 = None
    mul_211: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_153);  primals_153 = None
    unsqueeze_425: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_211, 0);  mul_211 = None
    unsqueeze_426: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_212: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_140, unsqueeze_427);  unsqueeze_427 = None
    mul_213: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_4);  sum_15 = rsqrt_4 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_212, mul_146, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_212 = mul_146 = primals_152 = None
    getitem_18: "f32[4, 960, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[160, 960, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_214: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, div_23);  div_23 = None
    mul_215: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, div_22);  getitem_18 = div_22 = None
    sum_16: "f32[4, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2, 3], True);  mul_215 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_1: "b8[4, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_54, -3.0)
    lt_5: "b8[4, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_54, 3.0);  convolution_54 = None
    bitwise_and_1: "b8[4, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_5);  gt_1 = lt_5 = None
    mul_216: "f32[4, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_16, 0.16666666666666666);  sum_16 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_216, scalar_tensor_6);  bitwise_and_1 = mul_216 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_17: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_10, relu_17, primals_150, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = relu_17 = primals_150 = None
    getitem_21: "f32[4, 240, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[960, 240, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_20: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    le_5: "b8[4, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_20, 0);  alias_20 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 240, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_7, getitem_21);  le_5 = scalar_tensor_7 = getitem_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_18: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, mean_6, primals_148, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean_6 = primals_148 = None
    getitem_24: "f32[4, 960, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[240, 960, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_2: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_24, [4, 960, 7, 7]);  getitem_24 = None
    div_35: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_142: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_214, div_35);  mul_214 = div_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_6: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_6: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_36: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_143: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_36, 0.5);  div_36 = None
    mul_217: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, add_143);  add_143 = None
    where_12: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_6, mul_217, add_142);  le_6 = mul_217 = add_142 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_6, scalar_tensor_8, where_12);  lt_6 = scalar_tensor_8 = where_12 = None
    add_144: "f32[960]" = torch.ops.aten.add.Tensor(primals_296, 0.001);  primals_296 = None
    rsqrt_5: "f32[960]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_428: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_295, 0);  primals_295 = None
    unsqueeze_429: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_19: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_51: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_430);  convolution_52 = unsqueeze_430 = None
    mul_218: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_51);  sub_51 = None
    sum_20: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 2, 3]);  mul_218 = None
    mul_223: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_146);  primals_146 = None
    unsqueeze_437: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_223, 0);  mul_223 = None
    unsqueeze_438: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_224: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_439);  where_13 = unsqueeze_439 = None
    mul_225: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, rsqrt_5);  sum_20 = rsqrt_5 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_224, div_21, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_224 = div_21 = primals_145 = None
    getitem_27: "f32[4, 960, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[960, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    lt_7: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_7: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_37: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_145: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
    mul_226: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_27, add_145);  add_145 = None
    where_14: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_7, mul_226, getitem_27);  le_7 = mul_226 = getitem_27 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_7, scalar_tensor_9, where_14);  lt_7 = scalar_tensor_9 = where_14 = None
    add_146: "f32[960]" = torch.ops.aten.add.Tensor(primals_293, 0.001);  primals_293 = None
    rsqrt_6: "f32[960]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_440: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_292, 0);  primals_292 = None
    unsqueeze_441: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_21: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_52: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_442);  convolution_51 = unsqueeze_442 = None
    mul_227: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_52);  sub_52 = None
    sum_22: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 2, 3]);  mul_227 = None
    mul_232: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_143);  primals_143 = None
    unsqueeze_449: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_232, 0);  mul_232 = None
    unsqueeze_450: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_233: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_451);  where_15 = unsqueeze_451 = None
    mul_234: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_6);  sum_22 = rsqrt_6 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_233, add_106, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_233 = add_106 = primals_142 = None
    getitem_30: "f32[4, 160, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[960, 160, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_147: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_140, getitem_30);  add_140 = getitem_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_148: "f32[160]" = torch.ops.aten.add.Tensor(primals_290, 0.001);  primals_290 = None
    rsqrt_7: "f32[160]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_452: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_289, 0);  primals_289 = None
    unsqueeze_453: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_23: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
    sub_53: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_454);  convolution_50 = unsqueeze_454 = None
    mul_235: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_147, sub_53);  sub_53 = None
    sum_24: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 2, 3]);  mul_235 = None
    mul_240: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_140);  primals_140 = None
    unsqueeze_461: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
    unsqueeze_462: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_241: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_147, unsqueeze_463);  add_147 = unsqueeze_463 = None
    mul_242: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_7);  sum_24 = rsqrt_7 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_241, mul_134, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = mul_134 = primals_139 = None
    getitem_33: "f32[4, 672, 7, 7]" = convolution_backward_11[0]
    getitem_34: "f32[160, 672, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_243: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, div_20);  div_20 = None
    mul_244: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, div_19);  getitem_33 = div_19 = None
    sum_25: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2, 3], True);  mul_244 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_2: "b8[4, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_49, -3.0)
    lt_8: "b8[4, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_49, 3.0);  convolution_49 = None
    bitwise_and_2: "b8[4, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_8);  gt_2 = lt_8 = None
    mul_245: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_25, 0.16666666666666666);  sum_25 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[4, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_245, scalar_tensor_10);  bitwise_and_2 = mul_245 = scalar_tensor_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_26: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_16, relu_16, primals_137, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = relu_16 = primals_137 = None
    getitem_36: "f32[4, 168, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[672, 168, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_21: "f32[4, 168, 1, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    le_8: "b8[4, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_21, 0);  alias_21 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[4, 168, 1, 1]" = torch.ops.aten.where.self(le_8, scalar_tensor_11, getitem_36);  le_8 = scalar_tensor_11 = getitem_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_27: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_17, mean_5, primals_135, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_5 = primals_135 = None
    getitem_39: "f32[4, 672, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[168, 672, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_3: "f32[4, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_39, [4, 672, 7, 7]);  getitem_39 = None
    div_38: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_149: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_243, div_38);  mul_243 = div_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_9: "b8[4, 672, 7, 7]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_9: "b8[4, 672, 7, 7]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_39: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_150: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
    mul_246: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_149, add_150);  add_150 = None
    where_18: "f32[4, 672, 7, 7]" = torch.ops.aten.where.self(le_9, mul_246, add_149);  le_9 = mul_246 = add_149 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[4, 672, 7, 7]" = torch.ops.aten.where.self(lt_9, scalar_tensor_12, where_18);  lt_9 = scalar_tensor_12 = where_18 = None
    add_151: "f32[672]" = torch.ops.aten.add.Tensor(primals_287, 0.001);  primals_287 = None
    rsqrt_8: "f32[672]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_464: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_286, 0);  primals_286 = None
    unsqueeze_465: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_28: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_54: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466);  convolution_47 = unsqueeze_466 = None
    mul_247: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_54);  sub_54 = None
    sum_29: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 2, 3]);  mul_247 = None
    mul_252: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_133);  primals_133 = None
    unsqueeze_473: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
    unsqueeze_474: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_253: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_475);  where_19 = unsqueeze_475 = None
    mul_254: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_8);  sum_29 = rsqrt_8 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_253, div_18, primals_132, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_253 = div_18 = primals_132 = None
    getitem_42: "f32[4, 672, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[672, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    lt_10: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_10: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_40: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_152: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
    mul_255: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_42, add_152);  add_152 = None
    where_20: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_10, mul_255, getitem_42);  le_10 = mul_255 = getitem_42 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_10, scalar_tensor_13, where_20);  lt_10 = scalar_tensor_13 = where_20 = None
    add_153: "f32[672]" = torch.ops.aten.add.Tensor(primals_284, 0.001);  primals_284 = None
    rsqrt_9: "f32[672]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_476: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_283, 0);  primals_283 = None
    unsqueeze_477: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_30: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_55: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478);  convolution_46 = unsqueeze_478 = None
    mul_256: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_55);  sub_55 = None
    sum_31: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 2, 3]);  mul_256 = None
    mul_261: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_130);  primals_130 = None
    unsqueeze_485: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
    unsqueeze_486: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_262: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_487);  where_21 = unsqueeze_487 = None
    mul_263: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_9);  sum_31 = rsqrt_9 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_262, add_97, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = add_97 = primals_129 = None
    getitem_45: "f32[4, 112, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[672, 112, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    add_154: "f32[112]" = torch.ops.aten.add.Tensor(primals_281, 0.001);  primals_281 = None
    rsqrt_10: "f32[112]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    unsqueeze_488: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_280, 0);  primals_280 = None
    unsqueeze_489: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_32: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_45, [0, 2, 3])
    sub_56: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490);  convolution_45 = unsqueeze_490 = None
    mul_264: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, sub_56);  sub_56 = None
    sum_33: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
    mul_269: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_127);  primals_127 = None
    unsqueeze_497: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_498: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_270: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_499);  unsqueeze_499 = None
    mul_271: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_10);  sum_33 = rsqrt_10 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_270, mul_122, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_270 = mul_122 = primals_126 = None
    getitem_48: "f32[4, 672, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[112, 672, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_272: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, div_17);  div_17 = None
    mul_273: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, div_16);  getitem_48 = div_16 = None
    sum_34: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2, 3], True);  mul_273 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_3: "b8[4, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_44, -3.0)
    lt_11: "b8[4, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_44, 3.0);  convolution_44 = None
    bitwise_and_3: "b8[4, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_11);  gt_3 = lt_11 = None
    mul_274: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[4, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_274, scalar_tensor_14);  bitwise_and_3 = mul_274 = scalar_tensor_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_22, relu_15, primals_124, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = relu_15 = primals_124 = None
    getitem_51: "f32[4, 168, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[672, 168, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_22: "f32[4, 168, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_11: "b8[4, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_22, 0);  alias_22 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[4, 168, 1, 1]" = torch.ops.aten.where.self(le_11, scalar_tensor_15, getitem_51);  le_11 = scalar_tensor_15 = getitem_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_36: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_23, mean_4, primals_122, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_4 = primals_122 = None
    getitem_54: "f32[4, 672, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[168, 672, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_4: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_54, [4, 672, 14, 14]);  getitem_54 = None
    div_41: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_155: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, div_41);  mul_272 = div_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_12: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_12: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_42: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_156: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
    mul_275: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_155, add_156);  add_156 = None
    where_24: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_12, mul_275, add_155);  le_12 = mul_275 = add_155 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_12, scalar_tensor_16, where_24);  lt_12 = scalar_tensor_16 = where_24 = None
    add_157: "f32[672]" = torch.ops.aten.add.Tensor(primals_278, 0.001);  primals_278 = None
    rsqrt_11: "f32[672]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_500: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_277, 0);  primals_277 = None
    unsqueeze_501: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_57: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_502);  convolution_42 = unsqueeze_502 = None
    mul_276: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_57);  sub_57 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3]);  mul_276 = None
    mul_281: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_120);  primals_120 = None
    unsqueeze_509: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_510: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_282: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_511);  where_25 = unsqueeze_511 = None
    mul_283: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, rsqrt_11);  sum_38 = rsqrt_11 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_282, div_15, primals_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_282 = div_15 = primals_119 = None
    getitem_57: "f32[4, 672, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[672, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    lt_13: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_13: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_43: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_158: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
    mul_284: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_57, add_158);  add_158 = None
    where_26: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_13, mul_284, getitem_57);  le_13 = mul_284 = getitem_57 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_13, scalar_tensor_17, where_26);  lt_13 = scalar_tensor_17 = where_26 = None
    add_159: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 0.001);  primals_275 = None
    rsqrt_12: "f32[672]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_512: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_274, 0);  primals_274 = None
    unsqueeze_513: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_39: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_58: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_514);  convolution_41 = unsqueeze_514 = None
    mul_285: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_58);  sub_58 = None
    sum_40: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3]);  mul_285 = None
    mul_290: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_117);  primals_117 = None
    unsqueeze_521: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_522: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_291: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_523);  where_27 = unsqueeze_523 = None
    mul_292: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, rsqrt_12);  sum_40 = rsqrt_12 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_291, add_87, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = add_87 = primals_116 = None
    getitem_60: "f32[4, 112, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[672, 112, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_160: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_45, getitem_60);  getitem_45 = getitem_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_161: "f32[112]" = torch.ops.aten.add.Tensor(primals_272, 0.001);  primals_272 = None
    rsqrt_13: "f32[112]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_524: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_271, 0);  primals_271 = None
    unsqueeze_525: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_41: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_160, [0, 2, 3])
    sub_59: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_526);  convolution_40 = unsqueeze_526 = None
    mul_293: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_160, sub_59);  sub_59 = None
    sum_42: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
    mul_298: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_114);  primals_114 = None
    unsqueeze_533: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_534: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_299: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_160, unsqueeze_535);  add_160 = unsqueeze_535 = None
    mul_300: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, rsqrt_13);  sum_42 = rsqrt_13 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_299, mul_110, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_299 = mul_110 = primals_113 = None
    getitem_63: "f32[4, 480, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[112, 480, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_301: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, div_14);  div_14 = None
    mul_302: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, div_13);  getitem_63 = div_13 = None
    sum_43: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2, 3], True);  mul_302 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_4: "b8[4, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_39, -3.0)
    lt_14: "b8[4, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, 3.0);  convolution_39 = None
    bitwise_and_4: "b8[4, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_14);  gt_4 = lt_14 = None
    mul_303: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_43, 0.16666666666666666);  sum_43 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[4, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_303, scalar_tensor_18);  bitwise_and_4 = mul_303 = scalar_tensor_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_28, relu_14, primals_111, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_28 = relu_14 = primals_111 = None
    getitem_66: "f32[4, 120, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[480, 120, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_23: "f32[4, 120, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    le_14: "b8[4, 120, 1, 1]" = torch.ops.aten.le.Scalar(alias_23, 0);  alias_23 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(le_14, scalar_tensor_19, getitem_66);  le_14 = scalar_tensor_19 = getitem_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_45: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_29, mean_3, primals_109, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_3 = primals_109 = None
    getitem_69: "f32[4, 480, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[120, 480, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_5: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_69, [4, 480, 14, 14]);  getitem_69 = None
    div_44: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_162: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_301, div_44);  mul_301 = div_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_15: "b8[4, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_15: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_45: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_163: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
    mul_304: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, add_163);  add_163 = None
    where_30: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_15, mul_304, add_162);  le_15 = mul_304 = add_162 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(lt_15, scalar_tensor_20, where_30);  lt_15 = scalar_tensor_20 = where_30 = None
    add_164: "f32[480]" = torch.ops.aten.add.Tensor(primals_269, 0.001);  primals_269 = None
    rsqrt_14: "f32[480]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_536: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_268, 0);  primals_268 = None
    unsqueeze_537: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_46: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_60: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_538);  convolution_37 = unsqueeze_538 = None
    mul_305: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_60);  sub_60 = None
    sum_47: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 2, 3]);  mul_305 = None
    mul_310: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_107);  primals_107 = None
    unsqueeze_545: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
    unsqueeze_546: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_311: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_547);  where_31 = unsqueeze_547 = None
    mul_312: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_14);  sum_47 = rsqrt_14 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_311, div_12, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_311 = div_12 = primals_106 = None
    getitem_72: "f32[4, 480, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[480, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    lt_16: "b8[4, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_16: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_46: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_165: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
    mul_313: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_72, add_165);  add_165 = None
    where_32: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_16, mul_313, getitem_72);  le_16 = mul_313 = getitem_72 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(lt_16, scalar_tensor_21, where_32);  lt_16 = scalar_tensor_21 = where_32 = None
    add_166: "f32[480]" = torch.ops.aten.add.Tensor(primals_266, 0.001);  primals_266 = None
    rsqrt_15: "f32[480]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_548: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_265, 0);  primals_265 = None
    unsqueeze_549: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_61: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_550);  convolution_36 = unsqueeze_550 = None
    mul_314: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_61);  sub_61 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_319: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_104);  primals_104 = None
    unsqueeze_557: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_558: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_320: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_559);  where_33 = unsqueeze_559 = None
    mul_321: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_15);  sum_49 = rsqrt_15 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_320, add_78, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = add_78 = primals_103 = None
    getitem_75: "f32[4, 80, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[480, 80, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    add_167: "f32[80]" = torch.ops.aten.add.Tensor(primals_263, 0.001);  primals_263 = None
    rsqrt_16: "f32[80]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_560: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_262, 0);  primals_262 = None
    unsqueeze_561: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_50: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_75, [0, 2, 3])
    sub_62: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_562);  convolution_35 = unsqueeze_562 = None
    mul_322: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, sub_62);  sub_62 = None
    sum_51: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 2, 3]);  mul_322 = None
    mul_327: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_101);  primals_101 = None
    unsqueeze_569: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_570: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_328: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_571);  unsqueeze_571 = None
    mul_329: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_16);  sum_51 = rsqrt_16 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_328, div_11, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = div_11 = primals_100 = None
    getitem_78: "f32[4, 184, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[80, 184, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    lt_17: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_17: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_47: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_168: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
    mul_330: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, add_168);  add_168 = None
    where_34: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_17, mul_330, getitem_78);  le_17 = mul_330 = getitem_78 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_17, scalar_tensor_22, where_34);  lt_17 = scalar_tensor_22 = where_34 = None
    add_169: "f32[184]" = torch.ops.aten.add.Tensor(primals_260, 0.001);  primals_260 = None
    rsqrt_17: "f32[184]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_572: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_259, 0);  primals_259 = None
    unsqueeze_573: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_52: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_63: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_574);  convolution_34 = unsqueeze_574 = None
    mul_331: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_63);  sub_63 = None
    sum_53: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 2, 3]);  mul_331 = None
    mul_336: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_98);  primals_98 = None
    unsqueeze_581: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_336, 0);  mul_336 = None
    unsqueeze_582: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_337: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_583);  where_35 = unsqueeze_583 = None
    mul_338: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_17);  sum_53 = rsqrt_17 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_337, div_10, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_337 = div_10 = primals_97 = None
    getitem_81: "f32[4, 184, 14, 14]" = convolution_backward_27[0]
    getitem_82: "f32[184, 1, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    lt_18: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_18: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_48: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_170: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_339: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_81, add_170);  add_170 = None
    where_36: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_18, mul_339, getitem_81);  le_18 = mul_339 = getitem_81 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_18, scalar_tensor_23, where_36);  lt_18 = scalar_tensor_23 = where_36 = None
    add_171: "f32[184]" = torch.ops.aten.add.Tensor(primals_257, 0.001);  primals_257 = None
    rsqrt_18: "f32[184]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_584: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_256, 0);  primals_256 = None
    unsqueeze_585: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_54: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_64: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_586);  convolution_33 = unsqueeze_586 = None
    mul_340: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_64);  sub_64 = None
    sum_55: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 2, 3]);  mul_340 = None
    mul_345: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_95);  primals_95 = None
    unsqueeze_593: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_594: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_346: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_595);  where_37 = unsqueeze_595 = None
    mul_347: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_18);  sum_55 = rsqrt_18 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_346, add_69, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_346 = add_69 = primals_94 = None
    getitem_84: "f32[4, 80, 14, 14]" = convolution_backward_28[0]
    getitem_85: "f32[184, 80, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_172: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_75, getitem_84);  getitem_75 = getitem_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_173: "f32[80]" = torch.ops.aten.add.Tensor(primals_254, 0.001);  primals_254 = None
    rsqrt_19: "f32[80]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_596: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_253, 0);  primals_253 = None
    unsqueeze_597: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_56: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 2, 3])
    sub_65: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_598);  convolution_32 = unsqueeze_598 = None
    mul_348: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, sub_65);  sub_65 = None
    sum_57: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_353: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_92);  primals_92 = None
    unsqueeze_605: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_606: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_354: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, unsqueeze_607);  unsqueeze_607 = None
    mul_355: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_19);  sum_57 = rsqrt_19 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_354, div_9, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = div_9 = primals_91 = None
    getitem_87: "f32[4, 184, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[80, 184, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    lt_19: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_19: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_49: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_174: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_356: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, add_174);  add_174 = None
    where_38: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_19, mul_356, getitem_87);  le_19 = mul_356 = getitem_87 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_19, scalar_tensor_24, where_38);  lt_19 = scalar_tensor_24 = where_38 = None
    add_175: "f32[184]" = torch.ops.aten.add.Tensor(primals_251, 0.001);  primals_251 = None
    rsqrt_20: "f32[184]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_608: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_250, 0);  primals_250 = None
    unsqueeze_609: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_58: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_66: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_610);  convolution_31 = unsqueeze_610 = None
    mul_357: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_66);  sub_66 = None
    sum_59: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_362: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_89);  primals_89 = None
    unsqueeze_617: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_618: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_363: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_619);  where_39 = unsqueeze_619 = None
    mul_364: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_20);  sum_59 = rsqrt_20 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_363, div_8, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_363 = div_8 = primals_88 = None
    getitem_90: "f32[4, 184, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[184, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    lt_20: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_20: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_50: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_176: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_365: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_90, add_176);  add_176 = None
    where_40: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_20, mul_365, getitem_90);  le_20 = mul_365 = getitem_90 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_20, scalar_tensor_25, where_40);  lt_20 = scalar_tensor_25 = where_40 = None
    add_177: "f32[184]" = torch.ops.aten.add.Tensor(primals_248, 0.001);  primals_248 = None
    rsqrt_21: "f32[184]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_620: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_247, 0);  primals_247 = None
    unsqueeze_621: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_60: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_67: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_622);  convolution_30 = unsqueeze_622 = None
    mul_366: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_67);  sub_67 = None
    sum_61: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
    mul_371: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_86);  primals_86 = None
    unsqueeze_629: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_630: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_372: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, unsqueeze_631);  where_41 = unsqueeze_631 = None
    mul_373: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_21);  sum_61 = rsqrt_21 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_372, add_60, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_372 = add_60 = primals_85 = None
    getitem_93: "f32[4, 80, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[184, 80, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_178: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_172, getitem_93);  add_172 = getitem_93 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_179: "f32[80]" = torch.ops.aten.add.Tensor(primals_245, 0.001);  primals_245 = None
    rsqrt_22: "f32[80]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_632: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_244, 0);  primals_244 = None
    unsqueeze_633: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_62: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 2, 3])
    sub_68: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_634);  convolution_29 = unsqueeze_634 = None
    mul_374: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_178, sub_68);  sub_68 = None
    sum_63: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3]);  mul_374 = None
    mul_379: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_83);  primals_83 = None
    unsqueeze_641: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_642: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_380: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_178, unsqueeze_643);  unsqueeze_643 = None
    mul_381: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_22);  sum_63 = rsqrt_22 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_380, div_7, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = div_7 = primals_82 = None
    getitem_96: "f32[4, 200, 14, 14]" = convolution_backward_32[0]
    getitem_97: "f32[80, 200, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    lt_21: "b8[4, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_21: "b8[4, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_51: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_180: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_382: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_96, add_180);  add_180 = None
    where_42: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(le_21, mul_382, getitem_96);  le_21 = mul_382 = getitem_96 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(lt_21, scalar_tensor_26, where_42);  lt_21 = scalar_tensor_26 = where_42 = None
    add_181: "f32[200]" = torch.ops.aten.add.Tensor(primals_242, 0.001);  primals_242 = None
    rsqrt_23: "f32[200]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_644: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(primals_241, 0);  primals_241 = None
    unsqueeze_645: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_64: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_69: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_646);  convolution_28 = unsqueeze_646 = None
    mul_383: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_69);  sub_69 = None
    sum_65: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_388: "f32[200]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_80);  primals_80 = None
    unsqueeze_653: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_654: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_389: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_655);  where_43 = unsqueeze_655 = None
    mul_390: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_23);  sum_65 = rsqrt_23 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_389, div_6, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False]);  mul_389 = div_6 = primals_79 = None
    getitem_99: "f32[4, 200, 14, 14]" = convolution_backward_33[0]
    getitem_100: "f32[200, 1, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    lt_22: "b8[4, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_22: "b8[4, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_52: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_182: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_391: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_99, add_182);  add_182 = None
    where_44: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(le_22, mul_391, getitem_99);  le_22 = mul_391 = getitem_99 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(lt_22, scalar_tensor_27, where_44);  lt_22 = scalar_tensor_27 = where_44 = None
    add_183: "f32[200]" = torch.ops.aten.add.Tensor(primals_239, 0.001);  primals_239 = None
    rsqrt_24: "f32[200]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    unsqueeze_656: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(primals_238, 0);  primals_238 = None
    unsqueeze_657: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_66: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_70: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_658);  convolution_27 = unsqueeze_658 = None
    mul_392: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_70);  sub_70 = None
    sum_67: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
    mul_397: "f32[200]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_77);  primals_77 = None
    unsqueeze_665: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
    unsqueeze_666: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_398: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_667);  where_45 = unsqueeze_667 = None
    mul_399: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_24);  sum_67 = rsqrt_24 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_398, add_51, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = add_51 = primals_76 = None
    getitem_102: "f32[4, 80, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[200, 80, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_184: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_178, getitem_102);  add_178 = getitem_102 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_185: "f32[80]" = torch.ops.aten.add.Tensor(primals_236, 0.001);  primals_236 = None
    rsqrt_25: "f32[80]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    unsqueeze_668: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_235, 0);  primals_235 = None
    unsqueeze_669: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_184, [0, 2, 3])
    sub_71: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_670);  convolution_26 = unsqueeze_670 = None
    mul_400: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, sub_71);  sub_71 = None
    sum_69: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_405: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_74);  primals_74 = None
    unsqueeze_677: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_678: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_406: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, unsqueeze_679);  add_184 = unsqueeze_679 = None
    mul_407: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_25);  sum_69 = rsqrt_25 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_406, div_5, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = div_5 = primals_73 = None
    getitem_105: "f32[4, 240, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[80, 240, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    lt_23: "b8[4, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_23: "b8[4, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_53: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_186: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_408: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_105, add_186);  add_186 = None
    where_46: "f32[4, 240, 14, 14]" = torch.ops.aten.where.self(le_23, mul_408, getitem_105);  le_23 = mul_408 = getitem_105 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[4, 240, 14, 14]" = torch.ops.aten.where.self(lt_23, scalar_tensor_28, where_46);  lt_23 = scalar_tensor_28 = where_46 = None
    add_187: "f32[240]" = torch.ops.aten.add.Tensor(primals_233, 0.001);  primals_233 = None
    rsqrt_26: "f32[240]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_680: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_232, 0);  primals_232 = None
    unsqueeze_681: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_70: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_72: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_682);  convolution_25 = unsqueeze_682 = None
    mul_409: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_72);  sub_72 = None
    sum_71: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_414: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_71);  primals_71 = None
    unsqueeze_689: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_690: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_415: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_691);  where_47 = unsqueeze_691 = None
    mul_416: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_26);  sum_71 = rsqrt_26 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_415, div_4, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_415 = div_4 = primals_70 = None
    getitem_108: "f32[4, 240, 28, 28]" = convolution_backward_36[0]
    getitem_109: "f32[240, 1, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    lt_24: "b8[4, 240, 28, 28]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_24: "b8[4, 240, 28, 28]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_54: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_188: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_417: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_108, add_188);  add_188 = None
    where_48: "f32[4, 240, 28, 28]" = torch.ops.aten.where.self(le_24, mul_417, getitem_108);  le_24 = mul_417 = getitem_108 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[4, 240, 28, 28]" = torch.ops.aten.where.self(lt_24, scalar_tensor_29, where_48);  lt_24 = scalar_tensor_29 = where_48 = None
    add_189: "f32[240]" = torch.ops.aten.add.Tensor(primals_230, 0.001);  primals_230 = None
    rsqrt_27: "f32[240]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    unsqueeze_692: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_229, 0);  primals_229 = None
    unsqueeze_693: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_72: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_73: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_694);  convolution_24 = unsqueeze_694 = None
    mul_418: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, sub_73);  sub_73 = None
    sum_73: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_423: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_68);  primals_68 = None
    unsqueeze_701: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_702: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_424: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, unsqueeze_703);  where_49 = unsqueeze_703 = None
    mul_425: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_27);  sum_73 = rsqrt_27 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_424, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_424 = add_43 = primals_67 = None
    getitem_111: "f32[4, 40, 28, 28]" = convolution_backward_37[0]
    getitem_112: "f32[240, 40, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    add_190: "f32[40]" = torch.ops.aten.add.Tensor(primals_227, 0.001);  primals_227 = None
    rsqrt_28: "f32[40]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_704: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_226, 0);  primals_226 = None
    unsqueeze_705: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_74: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_111, [0, 2, 3])
    sub_74: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_706);  convolution_23 = unsqueeze_706 = None
    mul_426: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_111, sub_74);  sub_74 = None
    sum_75: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_431: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_65);  primals_65 = None
    unsqueeze_713: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_714: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_432: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_111, unsqueeze_715);  unsqueeze_715 = None
    mul_433: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_28);  sum_75 = rsqrt_28 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_432, mul_54, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_432 = mul_54 = primals_64 = None
    getitem_114: "f32[4, 120, 28, 28]" = convolution_backward_38[0]
    getitem_115: "f32[40, 120, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_434: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, div_3);  div_3 = None
    mul_435: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, relu_12);  getitem_114 = None
    sum_76: "f32[4, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2, 3], True);  mul_435 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_5: "b8[4, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_22, -3.0)
    lt_25: "b8[4, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_22, 3.0);  convolution_22 = None
    bitwise_and_5: "b8[4, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_25);  gt_5 = lt_25 = None
    mul_436: "f32[4, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, 0.16666666666666666);  sum_76 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_50: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_436, scalar_tensor_30);  bitwise_and_5 = mul_436 = scalar_tensor_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_77: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_50, relu_13, primals_62, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_50 = relu_13 = primals_62 = None
    getitem_117: "f32[4, 32, 1, 1]" = convolution_backward_39[0]
    getitem_118: "f32[120, 32, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_24: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    le_25: "b8[4, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_24, 0);  alias_24 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[4, 32, 1, 1]" = torch.ops.aten.where.self(le_25, scalar_tensor_31, getitem_117);  le_25 = scalar_tensor_31 = getitem_117 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_78: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(where_51, mean_2, primals_60, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_51 = mean_2 = primals_60 = None
    getitem_120: "f32[4, 120, 1, 1]" = convolution_backward_40[0]
    getitem_121: "f32[32, 120, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_6: "f32[4, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_120, [4, 120, 28, 28]);  getitem_120 = None
    div_55: "f32[4, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_6, 784);  expand_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_191: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_434, div_55);  mul_434 = div_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    alias_26: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_27: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le_26: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_32, add_191);  le_26 = scalar_tensor_32 = add_191 = None
    add_192: "f32[120]" = torch.ops.aten.add.Tensor(primals_224, 0.001);  primals_224 = None
    rsqrt_29: "f32[120]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_716: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_223, 0);  primals_223 = None
    unsqueeze_717: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_79: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_75: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_718);  convolution_20 = unsqueeze_718 = None
    mul_437: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_75);  sub_75 = None
    sum_80: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_58);  primals_58 = None
    unsqueeze_725: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_726: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_443: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, unsqueeze_727);  where_52 = unsqueeze_727 = None
    mul_444: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, rsqrt_29);  sum_80 = rsqrt_29 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_443, relu_11, primals_57, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_443 = primals_57 = None
    getitem_123: "f32[4, 120, 28, 28]" = convolution_backward_41[0]
    getitem_124: "f32[120, 1, 5, 5]" = convolution_backward_41[1];  convolution_backward_41 = None
    alias_29: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_30: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    le_27: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_30, 0);  alias_30 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_53: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_33, getitem_123);  le_27 = scalar_tensor_33 = getitem_123 = None
    add_193: "f32[120]" = torch.ops.aten.add.Tensor(primals_221, 0.001);  primals_221 = None
    rsqrt_30: "f32[120]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    unsqueeze_728: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_220, 0);  primals_220 = None
    unsqueeze_729: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_76: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_730);  convolution_19 = unsqueeze_730 = None
    mul_445: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_76);  sub_76 = None
    sum_82: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_450: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_55);  primals_55 = None
    unsqueeze_737: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_738: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_451: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, unsqueeze_739);  where_53 = unsqueeze_739 = None
    mul_452: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_30);  sum_82 = rsqrt_30 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_451, add_35, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_451 = add_35 = primals_54 = None
    getitem_126: "f32[4, 40, 28, 28]" = convolution_backward_42[0]
    getitem_127: "f32[120, 40, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_194: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_111, getitem_126);  getitem_111 = getitem_126 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_195: "f32[40]" = torch.ops.aten.add.Tensor(primals_218, 0.001);  primals_218 = None
    rsqrt_31: "f32[40]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    unsqueeze_740: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_217, 0);  primals_217 = None
    unsqueeze_741: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_83: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_194, [0, 2, 3])
    sub_77: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_742);  convolution_18 = unsqueeze_742 = None
    mul_453: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_194, sub_77);  sub_77 = None
    sum_84: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
    mul_458: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_52);  primals_52 = None
    unsqueeze_749: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_750: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_459: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_194, unsqueeze_751);  unsqueeze_751 = None
    mul_460: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, rsqrt_31);  sum_84 = rsqrt_31 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_459, mul_44, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_459 = mul_44 = primals_51 = None
    getitem_129: "f32[4, 120, 28, 28]" = convolution_backward_43[0]
    getitem_130: "f32[40, 120, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_461: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_129, div_2);  div_2 = None
    mul_462: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_129, relu_9);  getitem_129 = None
    sum_85: "f32[4, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2, 3], True);  mul_462 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_6: "b8[4, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_17, -3.0)
    lt_26: "b8[4, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_17, 3.0);  convolution_17 = None
    bitwise_and_6: "b8[4, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_26);  gt_6 = lt_26 = None
    mul_463: "f32[4, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_85, 0.16666666666666666);  sum_85 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_54: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_463, scalar_tensor_34);  bitwise_and_6 = mul_463 = scalar_tensor_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_86: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_54, relu_10, primals_49, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_54 = relu_10 = primals_49 = None
    getitem_132: "f32[4, 32, 1, 1]" = convolution_backward_44[0]
    getitem_133: "f32[120, 32, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_31: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    le_28: "b8[4, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[4, 32, 1, 1]" = torch.ops.aten.where.self(le_28, scalar_tensor_35, getitem_132);  le_28 = scalar_tensor_35 = getitem_132 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_87: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_55, mean_1, primals_47, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_55 = mean_1 = primals_47 = None
    getitem_135: "f32[4, 120, 1, 1]" = convolution_backward_45[0]
    getitem_136: "f32[32, 120, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_7: "f32[4, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_135, [4, 120, 28, 28]);  getitem_135 = None
    div_56: "f32[4, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_196: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_461, div_56);  mul_461 = div_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    alias_33: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_34: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    le_29: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_34, 0);  alias_34 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_56: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_36, add_196);  le_29 = scalar_tensor_36 = add_196 = None
    add_197: "f32[120]" = torch.ops.aten.add.Tensor(primals_215, 0.001);  primals_215 = None
    rsqrt_32: "f32[120]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_752: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_214, 0);  primals_214 = None
    unsqueeze_753: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_88: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_78: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_754);  convolution_15 = unsqueeze_754 = None
    mul_464: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_78);  sub_78 = None
    sum_89: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_469: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_45);  primals_45 = None
    unsqueeze_761: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_762: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_470: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, unsqueeze_763);  where_56 = unsqueeze_763 = None
    mul_471: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_32);  sum_89 = rsqrt_32 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_470, relu_8, primals_44, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_470 = primals_44 = None
    getitem_138: "f32[4, 120, 28, 28]" = convolution_backward_46[0]
    getitem_139: "f32[120, 1, 5, 5]" = convolution_backward_46[1];  convolution_backward_46 = None
    alias_36: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_37: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_30: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_30, scalar_tensor_37, getitem_138);  le_30 = scalar_tensor_37 = getitem_138 = None
    add_198: "f32[120]" = torch.ops.aten.add.Tensor(primals_212, 0.001);  primals_212 = None
    rsqrt_33: "f32[120]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    unsqueeze_764: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_211, 0);  primals_211 = None
    unsqueeze_765: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_90: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_79: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_766);  convolution_14 = unsqueeze_766 = None
    mul_472: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_79);  sub_79 = None
    sum_91: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_477: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_42);  primals_42 = None
    unsqueeze_773: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_774: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_478: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, unsqueeze_775);  where_57 = unsqueeze_775 = None
    mul_479: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_33);  sum_91 = rsqrt_33 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_478, add_27, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_478 = add_27 = primals_41 = None
    getitem_141: "f32[4, 40, 28, 28]" = convolution_backward_47[0]
    getitem_142: "f32[120, 40, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_199: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_194, getitem_141);  add_194 = getitem_141 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_200: "f32[40]" = torch.ops.aten.add.Tensor(primals_209, 0.001);  primals_209 = None
    rsqrt_34: "f32[40]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    unsqueeze_776: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_208, 0);  primals_208 = None
    unsqueeze_777: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_92: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_199, [0, 2, 3])
    sub_80: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_778);  convolution_13 = unsqueeze_778 = None
    mul_480: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_199, sub_80);  sub_80 = None
    sum_93: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
    mul_485: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_39);  primals_39 = None
    unsqueeze_785: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_786: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_486: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_199, unsqueeze_787);  add_199 = unsqueeze_787 = None
    mul_487: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_34);  sum_93 = rsqrt_34 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_486, mul_34, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_34 = primals_38 = None
    getitem_144: "f32[4, 72, 28, 28]" = convolution_backward_48[0]
    getitem_145: "f32[40, 72, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_488: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_144, div_1);  div_1 = None
    mul_489: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_144, relu_6);  getitem_144 = None
    sum_94: "f32[4, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [2, 3], True);  mul_489 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt_7: "b8[4, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_12, -3.0)
    lt_27: "b8[4, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_12, 3.0);  convolution_12 = None
    bitwise_and_7: "b8[4, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_27);  gt_7 = lt_27 = None
    mul_490: "f32[4, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_94, 0.16666666666666666);  sum_94 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_58: "f32[4, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_490, scalar_tensor_38);  bitwise_and_7 = mul_490 = scalar_tensor_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_58, relu_7, primals_36, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_58 = relu_7 = primals_36 = None
    getitem_147: "f32[4, 24, 1, 1]" = convolution_backward_49[0]
    getitem_148: "f32[72, 24, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    alias_38: "f32[4, 24, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    le_31: "b8[4, 24, 1, 1]" = torch.ops.aten.le.Scalar(alias_38, 0);  alias_38 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[4, 24, 1, 1]" = torch.ops.aten.where.self(le_31, scalar_tensor_39, getitem_147);  le_31 = scalar_tensor_39 = getitem_147 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(where_59, mean, primals_34, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_59 = mean = primals_34 = None
    getitem_150: "f32[4, 72, 1, 1]" = convolution_backward_50[0]
    getitem_151: "f32[24, 72, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_8: "f32[4, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_150, [4, 72, 28, 28]);  getitem_150 = None
    div_57: "f32[4, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_8, 784);  expand_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_201: "f32[4, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_488, div_57);  mul_488 = div_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    alias_40: "f32[4, 72, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_41: "f32[4, 72, 28, 28]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    le_32: "b8[4, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_41, 0);  alias_41 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_60: "f32[4, 72, 28, 28]" = torch.ops.aten.where.self(le_32, scalar_tensor_40, add_201);  le_32 = scalar_tensor_40 = add_201 = None
    add_202: "f32[72]" = torch.ops.aten.add.Tensor(primals_206, 0.001);  primals_206 = None
    rsqrt_35: "f32[72]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    unsqueeze_788: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_205, 0);  primals_205 = None
    unsqueeze_789: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_97: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_81: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_790);  convolution_10 = unsqueeze_790 = None
    mul_491: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_81);  sub_81 = None
    sum_98: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_496: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_32);  primals_32 = None
    unsqueeze_797: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_798: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_497: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, unsqueeze_799);  where_60 = unsqueeze_799 = None
    mul_498: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, rsqrt_35);  sum_98 = rsqrt_35 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_497, relu_5, primals_31, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_497 = primals_31 = None
    getitem_153: "f32[4, 72, 56, 56]" = convolution_backward_51[0]
    getitem_154: "f32[72, 1, 5, 5]" = convolution_backward_51[1];  convolution_backward_51 = None
    alias_43: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_44: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_33: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_33, scalar_tensor_41, getitem_153);  le_33 = scalar_tensor_41 = getitem_153 = None
    add_203: "f32[72]" = torch.ops.aten.add.Tensor(primals_203, 0.001);  primals_203 = None
    rsqrt_36: "f32[72]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    unsqueeze_800: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_202, 0);  primals_202 = None
    unsqueeze_801: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_99: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_82: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_802);  convolution_9 = unsqueeze_802 = None
    mul_499: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_61, sub_82);  sub_82 = None
    sum_100: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_504: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_29);  primals_29 = None
    unsqueeze_809: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_810: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_505: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_61, unsqueeze_811);  where_61 = unsqueeze_811 = None
    mul_506: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, rsqrt_36);  sum_100 = rsqrt_36 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_505, add_20, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = add_20 = primals_28 = None
    getitem_156: "f32[4, 24, 56, 56]" = convolution_backward_52[0]
    getitem_157: "f32[72, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    add_204: "f32[24]" = torch.ops.aten.add.Tensor(primals_200, 0.001);  primals_200 = None
    rsqrt_37: "f32[24]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    unsqueeze_812: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_199, 0);  primals_199 = None
    unsqueeze_813: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_101: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_156, [0, 2, 3])
    sub_83: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_814);  convolution_8 = unsqueeze_814 = None
    mul_507: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_156, sub_83);  sub_83 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_512: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_26);  primals_26 = None
    unsqueeze_821: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_822: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_513: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_156, unsqueeze_823);  unsqueeze_823 = None
    mul_514: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, rsqrt_37);  sum_102 = rsqrt_37 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_513, relu_4, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_513 = primals_25 = None
    getitem_159: "f32[4, 72, 56, 56]" = convolution_backward_53[0]
    getitem_160: "f32[24, 72, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    alias_46: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_47: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_34: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_62: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_42, getitem_159);  le_34 = scalar_tensor_42 = getitem_159 = None
    add_205: "f32[72]" = torch.ops.aten.add.Tensor(primals_197, 0.001);  primals_197 = None
    rsqrt_38: "f32[72]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_824: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_196, 0);  primals_196 = None
    unsqueeze_825: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_103: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_84: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_826);  convolution_7 = unsqueeze_826 = None
    mul_515: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_62, sub_84);  sub_84 = None
    sum_104: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
    mul_520: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_23);  primals_23 = None
    unsqueeze_833: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_834: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_521: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_62, unsqueeze_835);  where_62 = unsqueeze_835 = None
    mul_522: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, rsqrt_38);  sum_104 = rsqrt_38 = None
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_521, relu_3, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_521 = primals_22 = None
    getitem_162: "f32[4, 72, 56, 56]" = convolution_backward_54[0]
    getitem_163: "f32[72, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    alias_49: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_50: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_35: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_35, scalar_tensor_43, getitem_162);  le_35 = scalar_tensor_43 = getitem_162 = None
    add_206: "f32[72]" = torch.ops.aten.add.Tensor(primals_194, 0.001);  primals_194 = None
    rsqrt_39: "f32[72]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    unsqueeze_836: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_193, 0);  primals_193 = None
    unsqueeze_837: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_105: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_85: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_838);  convolution_6 = unsqueeze_838 = None
    mul_523: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_63, sub_85);  sub_85 = None
    sum_106: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 2, 3]);  mul_523 = None
    mul_528: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_20);  primals_20 = None
    unsqueeze_845: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_846: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_529: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_63, unsqueeze_847);  where_63 = unsqueeze_847 = None
    mul_530: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, rsqrt_39);  sum_106 = rsqrt_39 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_529, add_13, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_529 = add_13 = primals_19 = None
    getitem_165: "f32[4, 24, 56, 56]" = convolution_backward_55[0]
    getitem_166: "f32[72, 24, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_207: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_156, getitem_165);  getitem_156 = getitem_165 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_208: "f32[24]" = torch.ops.aten.add.Tensor(primals_191, 0.001);  primals_191 = None
    rsqrt_40: "f32[24]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_848: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_190, 0);  primals_190 = None
    unsqueeze_849: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_207, [0, 2, 3])
    sub_86: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_850);  convolution_5 = unsqueeze_850 = None
    mul_531: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_207, sub_86);  sub_86 = None
    sum_108: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None
    mul_536: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_17);  primals_17 = None
    unsqueeze_857: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_858: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_537: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_207, unsqueeze_859);  add_207 = unsqueeze_859 = None
    mul_538: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, rsqrt_40);  sum_108 = rsqrt_40 = None
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_537, relu_2, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_537 = primals_16 = None
    getitem_168: "f32[4, 64, 56, 56]" = convolution_backward_56[0]
    getitem_169: "f32[24, 64, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    alias_52: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_53: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_36: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_53, 0);  alias_53 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_36, scalar_tensor_44, getitem_168);  le_36 = scalar_tensor_44 = getitem_168 = None
    add_209: "f32[64]" = torch.ops.aten.add.Tensor(primals_188, 0.001);  primals_188 = None
    rsqrt_41: "f32[64]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    unsqueeze_860: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_187, 0);  primals_187 = None
    unsqueeze_861: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_87: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_862);  convolution_4 = unsqueeze_862 = None
    mul_539: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, sub_87);  sub_87 = None
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
    mul_544: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_14);  primals_14 = None
    unsqueeze_869: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_870: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_545: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_871);  where_64 = unsqueeze_871 = None
    mul_546: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, rsqrt_41);  sum_110 = rsqrt_41 = None
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_545, relu_1, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_545 = primals_13 = None
    getitem_171: "f32[4, 64, 112, 112]" = convolution_backward_57[0]
    getitem_172: "f32[64, 1, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    alias_55: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_56: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_37: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_65: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_37, scalar_tensor_45, getitem_171);  le_37 = scalar_tensor_45 = getitem_171 = None
    add_210: "f32[64]" = torch.ops.aten.add.Tensor(primals_185, 0.001);  primals_185 = None
    rsqrt_42: "f32[64]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    unsqueeze_872: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_184, 0);  primals_184 = None
    unsqueeze_873: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_88: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_874);  convolution_3 = unsqueeze_874 = None
    mul_547: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_65, sub_88);  sub_88 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_552: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_11);  primals_11 = None
    unsqueeze_881: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_882: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_553: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_65, unsqueeze_883);  where_65 = unsqueeze_883 = None
    mul_554: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, rsqrt_42);  sum_112 = rsqrt_42 = None
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_553, add_7, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_553 = add_7 = primals_10 = None
    getitem_174: "f32[4, 16, 112, 112]" = convolution_backward_58[0]
    getitem_175: "f32[64, 16, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    add_211: "f32[16]" = torch.ops.aten.add.Tensor(primals_182, 0.001);  primals_182 = None
    rsqrt_43: "f32[16]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
    unsqueeze_884: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_181, 0);  primals_181 = None
    unsqueeze_885: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_113: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_174, [0, 2, 3])
    sub_89: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_886);  convolution_2 = unsqueeze_886 = None
    mul_555: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, sub_89);  sub_89 = None
    sum_114: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
    mul_560: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_8);  primals_8 = None
    unsqueeze_893: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_894: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_561: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, unsqueeze_895);  unsqueeze_895 = None
    mul_562: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, rsqrt_43);  sum_114 = rsqrt_43 = None
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_561, relu, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_561 = primals_7 = None
    getitem_177: "f32[4, 16, 112, 112]" = convolution_backward_59[0]
    getitem_178: "f32[16, 16, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    alias_58: "f32[4, 16, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_59: "f32[4, 16, 112, 112]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_38: "b8[4, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_66: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(le_38, scalar_tensor_46, getitem_177);  le_38 = scalar_tensor_46 = getitem_177 = None
    add_212: "f32[16]" = torch.ops.aten.add.Tensor(primals_179, 0.001);  primals_179 = None
    rsqrt_44: "f32[16]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_896: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_178, 0);  primals_178 = None
    unsqueeze_897: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_115: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_90: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_898);  convolution_1 = unsqueeze_898 = None
    mul_563: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_66, sub_90);  sub_90 = None
    sum_116: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_568: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_5);  primals_5 = None
    unsqueeze_905: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_906: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_569: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_66, unsqueeze_907);  where_66 = unsqueeze_907 = None
    mul_570: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, rsqrt_44);  sum_116 = rsqrt_44 = None
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_569, div, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_569 = div = primals_4 = None
    getitem_180: "f32[4, 16, 112, 112]" = convolution_backward_60[0]
    getitem_181: "f32[16, 1, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_213: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_174, getitem_180);  getitem_174 = getitem_180 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    lt_28: "b8[4, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_39: "b8[4, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_58: "f32[4, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_214: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_571: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_213, add_214);  add_214 = None
    where_67: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(le_39, mul_571, add_213);  le_39 = mul_571 = add_213 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(lt_28, scalar_tensor_47, where_67);  lt_28 = scalar_tensor_47 = where_67 = None
    add_215: "f32[16]" = torch.ops.aten.add.Tensor(primals_176, 0.001);  primals_176 = None
    rsqrt_45: "f32[16]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    unsqueeze_908: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_175, 0);  primals_175 = None
    unsqueeze_909: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_117: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_91: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_910);  convolution = unsqueeze_910 = None
    mul_572: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_68, sub_91);  sub_91 = None
    sum_118: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
    mul_577: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_2);  primals_2 = None
    unsqueeze_917: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_918: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_578: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_68, unsqueeze_919);  where_68 = unsqueeze_919 = None
    mul_579: "f32[16]" = torch.ops.aten.mul.Tensor(sum_118, rsqrt_45);  sum_118 = rsqrt_45 = None
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_578, primals_313, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_578 = primals_313 = primals_1 = None
    getitem_184: "f32[16, 3, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    return pytree.tree_unflatten([addmm_1, getitem_184, mul_579, sum_117, getitem_181, mul_570, sum_115, getitem_178, mul_562, sum_113, getitem_175, mul_554, sum_111, getitem_172, mul_546, sum_109, getitem_169, mul_538, sum_107, getitem_166, mul_530, sum_105, getitem_163, mul_522, sum_103, getitem_160, mul_514, sum_101, getitem_157, mul_506, sum_99, getitem_154, mul_498, sum_97, getitem_151, sum_96, getitem_148, sum_95, getitem_145, mul_487, sum_92, getitem_142, mul_479, sum_90, getitem_139, mul_471, sum_88, getitem_136, sum_87, getitem_133, sum_86, getitem_130, mul_460, sum_83, getitem_127, mul_452, sum_81, getitem_124, mul_444, sum_79, getitem_121, sum_78, getitem_118, sum_77, getitem_115, mul_433, sum_74, getitem_112, mul_425, sum_72, getitem_109, mul_416, sum_70, getitem_106, mul_407, sum_68, getitem_103, mul_399, sum_66, getitem_100, mul_390, sum_64, getitem_97, mul_381, sum_62, getitem_94, mul_373, sum_60, getitem_91, mul_364, sum_58, getitem_88, mul_355, sum_56, getitem_85, mul_347, sum_54, getitem_82, mul_338, sum_52, getitem_79, mul_329, sum_50, getitem_76, mul_321, sum_48, getitem_73, mul_312, sum_46, getitem_70, sum_45, getitem_67, sum_44, getitem_64, mul_300, sum_41, getitem_61, mul_292, sum_39, getitem_58, mul_283, sum_37, getitem_55, sum_36, getitem_52, sum_35, getitem_49, mul_271, sum_32, getitem_46, mul_263, sum_30, getitem_43, mul_254, sum_28, getitem_40, sum_27, getitem_37, sum_26, getitem_34, mul_242, sum_23, getitem_31, mul_234, sum_21, getitem_28, mul_225, sum_19, getitem_25, sum_18, getitem_22, sum_17, getitem_19, mul_213, sum_14, getitem_16, mul_205, sum_12, getitem_13, mul_196, sum_10, getitem_10, sum_9, getitem_7, sum_8, getitem_4, mul_184, sum_5, getitem_1, mul_176, sum_3, permute_9, view_2, permute_5, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    