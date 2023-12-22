from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32, 3, 3, 3]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[32, 1, 3, 3]"; primals_5: "f32[32]"; primals_6: "f32[32]"; primals_7: "f32[16, 32, 1, 1]"; primals_8: "f32[16]"; primals_9: "f32[16]"; primals_10: "f32[48, 16, 1, 1]"; primals_11: "f32[48]"; primals_12: "f32[48]"; primals_13: "f32[48, 1, 3, 3]"; primals_14: "f32[48]"; primals_15: "f32[48]"; primals_16: "f32[24, 48, 1, 1]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[72, 24, 1, 1]"; primals_20: "f32[72]"; primals_21: "f32[72]"; primals_22: "f32[72, 1, 3, 3]"; primals_23: "f32[72]"; primals_24: "f32[72]"; primals_25: "f32[24, 72, 1, 1]"; primals_26: "f32[24]"; primals_27: "f32[24]"; primals_28: "f32[72, 24, 1, 1]"; primals_29: "f32[72]"; primals_30: "f32[72]"; primals_31: "f32[72, 1, 3, 3]"; primals_32: "f32[72]"; primals_33: "f32[72]"; primals_34: "f32[24, 72, 1, 1]"; primals_35: "f32[24]"; primals_36: "f32[24]"; primals_37: "f32[72, 24, 1, 1]"; primals_38: "f32[72]"; primals_39: "f32[72]"; primals_40: "f32[72, 1, 5, 5]"; primals_41: "f32[72]"; primals_42: "f32[72]"; primals_43: "f32[40, 72, 1, 1]"; primals_44: "f32[40]"; primals_45: "f32[40]"; primals_46: "f32[120, 40, 1, 1]"; primals_47: "f32[120]"; primals_48: "f32[120]"; primals_49: "f32[120, 1, 5, 5]"; primals_50: "f32[120]"; primals_51: "f32[120]"; primals_52: "f32[40, 120, 1, 1]"; primals_53: "f32[40]"; primals_54: "f32[40]"; primals_55: "f32[120, 40, 1, 1]"; primals_56: "f32[120]"; primals_57: "f32[120]"; primals_58: "f32[120, 1, 5, 5]"; primals_59: "f32[120]"; primals_60: "f32[120]"; primals_61: "f32[40, 120, 1, 1]"; primals_62: "f32[40]"; primals_63: "f32[40]"; primals_64: "f32[240, 40, 1, 1]"; primals_65: "f32[240]"; primals_66: "f32[240]"; primals_67: "f32[240, 1, 5, 5]"; primals_68: "f32[240]"; primals_69: "f32[240]"; primals_70: "f32[80, 240, 1, 1]"; primals_71: "f32[80]"; primals_72: "f32[80]"; primals_73: "f32[480, 80, 1, 1]"; primals_74: "f32[480]"; primals_75: "f32[480]"; primals_76: "f32[480, 1, 5, 5]"; primals_77: "f32[480]"; primals_78: "f32[480]"; primals_79: "f32[80, 480, 1, 1]"; primals_80: "f32[80]"; primals_81: "f32[80]"; primals_82: "f32[480, 80, 1, 1]"; primals_83: "f32[480]"; primals_84: "f32[480]"; primals_85: "f32[480, 1, 5, 5]"; primals_86: "f32[480]"; primals_87: "f32[480]"; primals_88: "f32[80, 480, 1, 1]"; primals_89: "f32[80]"; primals_90: "f32[80]"; primals_91: "f32[480, 80, 1, 1]"; primals_92: "f32[480]"; primals_93: "f32[480]"; primals_94: "f32[480, 1, 3, 3]"; primals_95: "f32[480]"; primals_96: "f32[480]"; primals_97: "f32[96, 480, 1, 1]"; primals_98: "f32[96]"; primals_99: "f32[96]"; primals_100: "f32[576, 96, 1, 1]"; primals_101: "f32[576]"; primals_102: "f32[576]"; primals_103: "f32[576, 1, 3, 3]"; primals_104: "f32[576]"; primals_105: "f32[576]"; primals_106: "f32[96, 576, 1, 1]"; primals_107: "f32[96]"; primals_108: "f32[96]"; primals_109: "f32[576, 96, 1, 1]"; primals_110: "f32[576]"; primals_111: "f32[576]"; primals_112: "f32[576, 1, 5, 5]"; primals_113: "f32[576]"; primals_114: "f32[576]"; primals_115: "f32[192, 576, 1, 1]"; primals_116: "f32[192]"; primals_117: "f32[192]"; primals_118: "f32[1152, 192, 1, 1]"; primals_119: "f32[1152]"; primals_120: "f32[1152]"; primals_121: "f32[1152, 1, 5, 5]"; primals_122: "f32[1152]"; primals_123: "f32[1152]"; primals_124: "f32[192, 1152, 1, 1]"; primals_125: "f32[192]"; primals_126: "f32[192]"; primals_127: "f32[1152, 192, 1, 1]"; primals_128: "f32[1152]"; primals_129: "f32[1152]"; primals_130: "f32[1152, 1, 5, 5]"; primals_131: "f32[1152]"; primals_132: "f32[1152]"; primals_133: "f32[192, 1152, 1, 1]"; primals_134: "f32[192]"; primals_135: "f32[192]"; primals_136: "f32[1152, 192, 1, 1]"; primals_137: "f32[1152]"; primals_138: "f32[1152]"; primals_139: "f32[1152, 1, 5, 5]"; primals_140: "f32[1152]"; primals_141: "f32[1152]"; primals_142: "f32[192, 1152, 1, 1]"; primals_143: "f32[192]"; primals_144: "f32[192]"; primals_145: "f32[1152, 192, 1, 1]"; primals_146: "f32[1152]"; primals_147: "f32[1152]"; primals_148: "f32[1152, 1, 3, 3]"; primals_149: "f32[1152]"; primals_150: "f32[1152]"; primals_151: "f32[320, 1152, 1, 1]"; primals_152: "f32[320]"; primals_153: "f32[320]"; primals_154: "f32[1280, 320, 1, 1]"; primals_155: "f32[1280]"; primals_156: "f32[1280]"; primals_157: "f32[1000, 1280]"; primals_158: "f32[1000]"; primals_159: "f32[32]"; primals_160: "f32[32]"; primals_161: "i64[]"; primals_162: "f32[32]"; primals_163: "f32[32]"; primals_164: "i64[]"; primals_165: "f32[16]"; primals_166: "f32[16]"; primals_167: "i64[]"; primals_168: "f32[48]"; primals_169: "f32[48]"; primals_170: "i64[]"; primals_171: "f32[48]"; primals_172: "f32[48]"; primals_173: "i64[]"; primals_174: "f32[24]"; primals_175: "f32[24]"; primals_176: "i64[]"; primals_177: "f32[72]"; primals_178: "f32[72]"; primals_179: "i64[]"; primals_180: "f32[72]"; primals_181: "f32[72]"; primals_182: "i64[]"; primals_183: "f32[24]"; primals_184: "f32[24]"; primals_185: "i64[]"; primals_186: "f32[72]"; primals_187: "f32[72]"; primals_188: "i64[]"; primals_189: "f32[72]"; primals_190: "f32[72]"; primals_191: "i64[]"; primals_192: "f32[24]"; primals_193: "f32[24]"; primals_194: "i64[]"; primals_195: "f32[72]"; primals_196: "f32[72]"; primals_197: "i64[]"; primals_198: "f32[72]"; primals_199: "f32[72]"; primals_200: "i64[]"; primals_201: "f32[40]"; primals_202: "f32[40]"; primals_203: "i64[]"; primals_204: "f32[120]"; primals_205: "f32[120]"; primals_206: "i64[]"; primals_207: "f32[120]"; primals_208: "f32[120]"; primals_209: "i64[]"; primals_210: "f32[40]"; primals_211: "f32[40]"; primals_212: "i64[]"; primals_213: "f32[120]"; primals_214: "f32[120]"; primals_215: "i64[]"; primals_216: "f32[120]"; primals_217: "f32[120]"; primals_218: "i64[]"; primals_219: "f32[40]"; primals_220: "f32[40]"; primals_221: "i64[]"; primals_222: "f32[240]"; primals_223: "f32[240]"; primals_224: "i64[]"; primals_225: "f32[240]"; primals_226: "f32[240]"; primals_227: "i64[]"; primals_228: "f32[80]"; primals_229: "f32[80]"; primals_230: "i64[]"; primals_231: "f32[480]"; primals_232: "f32[480]"; primals_233: "i64[]"; primals_234: "f32[480]"; primals_235: "f32[480]"; primals_236: "i64[]"; primals_237: "f32[80]"; primals_238: "f32[80]"; primals_239: "i64[]"; primals_240: "f32[480]"; primals_241: "f32[480]"; primals_242: "i64[]"; primals_243: "f32[480]"; primals_244: "f32[480]"; primals_245: "i64[]"; primals_246: "f32[80]"; primals_247: "f32[80]"; primals_248: "i64[]"; primals_249: "f32[480]"; primals_250: "f32[480]"; primals_251: "i64[]"; primals_252: "f32[480]"; primals_253: "f32[480]"; primals_254: "i64[]"; primals_255: "f32[96]"; primals_256: "f32[96]"; primals_257: "i64[]"; primals_258: "f32[576]"; primals_259: "f32[576]"; primals_260: "i64[]"; primals_261: "f32[576]"; primals_262: "f32[576]"; primals_263: "i64[]"; primals_264: "f32[96]"; primals_265: "f32[96]"; primals_266: "i64[]"; primals_267: "f32[576]"; primals_268: "f32[576]"; primals_269: "i64[]"; primals_270: "f32[576]"; primals_271: "f32[576]"; primals_272: "i64[]"; primals_273: "f32[192]"; primals_274: "f32[192]"; primals_275: "i64[]"; primals_276: "f32[1152]"; primals_277: "f32[1152]"; primals_278: "i64[]"; primals_279: "f32[1152]"; primals_280: "f32[1152]"; primals_281: "i64[]"; primals_282: "f32[192]"; primals_283: "f32[192]"; primals_284: "i64[]"; primals_285: "f32[1152]"; primals_286: "f32[1152]"; primals_287: "i64[]"; primals_288: "f32[1152]"; primals_289: "f32[1152]"; primals_290: "i64[]"; primals_291: "f32[192]"; primals_292: "f32[192]"; primals_293: "i64[]"; primals_294: "f32[1152]"; primals_295: "f32[1152]"; primals_296: "i64[]"; primals_297: "f32[1152]"; primals_298: "f32[1152]"; primals_299: "i64[]"; primals_300: "f32[192]"; primals_301: "f32[192]"; primals_302: "i64[]"; primals_303: "f32[1152]"; primals_304: "f32[1152]"; primals_305: "i64[]"; primals_306: "f32[1152]"; primals_307: "f32[1152]"; primals_308: "i64[]"; primals_309: "f32[320]"; primals_310: "f32[320]"; primals_311: "i64[]"; primals_312: "f32[1280]"; primals_313: "f32[1280]"; primals_314: "i64[]"; primals_315: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:159, code: x = self.layers(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_315, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_159, torch.float32)
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_160, torch.float32)
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    convolution_1: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_162, torch.float32)
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_163, torch.float32)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    convolution_2: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_165, torch.float32)
    convert_element_type_5: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_166, torch.float32)
    add_4: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_3: "f32[4, 48, 112, 112]" = torch.ops.aten.convolution.default(add_5, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_6: "f32[48]" = torch.ops.prims.convert_element_type.default(primals_168, torch.float32)
    convert_element_type_7: "f32[48]" = torch.ops.prims.convert_element_type.default(primals_169, torch.float32)
    add_6: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[48]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 48, 112, 112]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 48, 112, 112]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    relu_2: "f32[4, 48, 112, 112]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    convolution_4: "f32[4, 48, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48)
    convert_element_type_8: "f32[48]" = torch.ops.prims.convert_element_type.default(primals_171, torch.float32)
    convert_element_type_9: "f32[48]" = torch.ops.prims.convert_element_type.default(primals_172, torch.float32)
    add_8: "f32[48]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[48]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    relu_3: "f32[4, 48, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_5: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_10: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_174, torch.float32)
    convert_element_type_11: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    add_10: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    convolution_6: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_12: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_177, torch.float32)
    convert_element_type_13: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    add_12: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[72]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_4: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    convolution_7: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    convert_element_type_14: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_180, torch.float32)
    convert_element_type_15: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    add_14: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[72]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    relu_5: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    convolution_8: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_16: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_183, torch.float32)
    convert_element_type_17: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    add_16: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    add_18: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_17, add_11);  add_17 = None
    convolution_9: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_18: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_186, torch.float32)
    convert_element_type_19: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    add_19: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[72]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    relu_6: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    convolution_10: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    convert_element_type_20: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_189, torch.float32)
    convert_element_type_21: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    add_21: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[72]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    relu_7: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    convolution_11: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_22: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_192, torch.float32)
    convert_element_type_23: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    add_23: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[24]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    add_25: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_24, add_18);  add_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_12: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_25, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_195, torch.float32)
    convert_element_type_25: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    add_26: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[72]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_12: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_27: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    relu_8: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    convolution_13: "f32[4, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_40, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    convert_element_type_26: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_198, torch.float32)
    convert_element_type_27: "f32[72]" = torch.ops.prims.convert_element_type.default(primals_199, torch.float32)
    add_28: "f32[72]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[72]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[4, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    relu_9: "f32[4, 72, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    convolution_14: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_201, torch.float32)
    convert_element_type_29: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_202, torch.float32)
    add_30: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[40]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    convolution_15: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_30: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_204, torch.float32)
    convert_element_type_31: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_205, torch.float32)
    add_32: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[120]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    relu_10: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    convolution_16: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_49, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    convert_element_type_32: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_207, torch.float32)
    convert_element_type_33: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_208, torch.float32)
    add_34: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[120]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_11: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    convolution_17: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_210, torch.float32)
    convert_element_type_35: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_211, torch.float32)
    add_36: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[40]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    add_38: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_37, add_31);  add_37 = None
    convolution_18: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_38, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_36: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_213, torch.float32)
    convert_element_type_37: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    add_39: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[120]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    relu_12: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    convolution_19: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_58, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    convert_element_type_38: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_216, torch.float32)
    convert_element_type_39: "f32[120]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    add_41: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[120]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_19: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_42: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    relu_13: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    convolution_20: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_40: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_219, torch.float32)
    convert_element_type_41: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    add_43: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[40]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_20: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_165: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_167: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_44: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    add_45: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_44, add_38);  add_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_21: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_45, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_42: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_222, torch.float32)
    convert_element_type_43: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    add_46: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[240]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_21: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_173: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_175: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_47: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    relu_14: "f32[4, 240, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    convolution_22: "f32[4, 240, 14, 14]" = torch.ops.aten.convolution.default(relu_14, primals_67, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 240)
    convert_element_type_44: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_225, torch.float32)
    convert_element_type_45: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    add_48: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[240]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_22: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_181: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_183: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_49: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    relu_15: "f32[4, 240, 14, 14]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    convolution_23: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_46: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_228, torch.float32)
    convert_element_type_47: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    add_50: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[80]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_23: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_189: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_191: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_51: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    convolution_24: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_48: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_231, torch.float32)
    convert_element_type_49: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    add_52: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[480]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_24: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_197: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_199: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_53: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    relu_16: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    convolution_25: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_76, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    convert_element_type_50: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_234, torch.float32)
    convert_element_type_51: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    add_54: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[480]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_25: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_205: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_207: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_55: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    relu_17: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    convolution_26: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_52: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_237, torch.float32)
    convert_element_type_53: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    add_56: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[80]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_26: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_213: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_215: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_57: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    add_58: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_57, add_51);  add_57 = None
    convolution_27: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_58, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_54: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_240, torch.float32)
    convert_element_type_55: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    add_59: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[480]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_27: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_221: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_223: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_60: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    relu_18: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    convolution_28: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_18, primals_85, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    convert_element_type_56: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_243, torch.float32)
    convert_element_type_57: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    add_61: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[480]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_28: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_229: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_231: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_62: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    relu_19: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_62);  add_62 = None
    convolution_29: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_19, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_58: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_246, torch.float32)
    convert_element_type_59: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    add_63: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[80]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_29: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_237: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_239: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_64: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    add_65: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_64, add_58);  add_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_30: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_65, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_60: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_249, torch.float32)
    convert_element_type_61: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    add_66: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[480]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_30: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_245: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_247: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_67: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    relu_20: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    convolution_31: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    convert_element_type_62: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_252, torch.float32)
    convert_element_type_63: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    add_68: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[480]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_31: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_253: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_255: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_69: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    relu_21: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    convolution_32: "f32[4, 96, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_64: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_255, torch.float32)
    convert_element_type_65: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    add_70: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[96]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_32: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_261: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_263: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_71: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    convolution_33: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(add_71, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_66: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_258, torch.float32)
    convert_element_type_67: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    add_72: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[576]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_33: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_269: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_271: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_73: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    relu_22: "f32[4, 576, 14, 14]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    convolution_34: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576)
    convert_element_type_68: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_261, torch.float32)
    convert_element_type_69: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    add_74: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[576]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_34: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_277: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_279: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_75: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    relu_23: "f32[4, 576, 14, 14]" = torch.ops.aten.relu.default(add_75);  add_75 = None
    convolution_35: "f32[4, 96, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_70: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_264, torch.float32)
    convert_element_type_71: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    add_76: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[96]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_35: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_285: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_287: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_77: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    add_78: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_77, add_71);  add_77 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_36: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(add_78, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_72: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    convert_element_type_73: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    add_79: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[576]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_36: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_293: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_295: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_80: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    relu_24: "f32[4, 576, 14, 14]" = torch.ops.aten.relu.default(add_80);  add_80 = None
    convolution_37: "f32[4, 576, 7, 7]" = torch.ops.aten.convolution.default(relu_24, primals_112, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 576)
    convert_element_type_74: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    convert_element_type_75: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    add_81: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[576]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_37: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_301: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_303: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_82: "f32[4, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    relu_25: "f32[4, 576, 7, 7]" = torch.ops.aten.relu.default(add_82);  add_82 = None
    convolution_38: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_25, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_76: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    convert_element_type_77: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    add_83: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[192]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
    reciprocal_38: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_309: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_311: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_84: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    convolution_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_84, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_78: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    convert_element_type_79: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    add_85: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[1152]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_39: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  unsqueeze_313 = None
    mul_118: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_317: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_319: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_86: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    relu_26: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    convolution_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_26, primals_121, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    convert_element_type_80: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    convert_element_type_81: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    add_87: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[1152]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_40: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  unsqueeze_321 = None
    mul_121: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_325: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_327: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_88: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    relu_27: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    convolution_41: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_27, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_82: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    convert_element_type_83: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    add_89: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[192]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_41: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  unsqueeze_329 = None
    mul_124: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_333: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_335: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_90: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    add_91: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_90, add_84);  add_90 = None
    convolution_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_91, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_84: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    convert_element_type_85: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    add_92: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[1152]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_42: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  unsqueeze_337 = None
    mul_127: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_341: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_343: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_93: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    relu_28: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    convolution_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_130, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    convert_element_type_86: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    convert_element_type_87: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    add_94: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[1152]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_43: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  unsqueeze_345 = None
    mul_130: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_349: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_351: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_95: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    relu_29: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    convolution_44: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_88: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    convert_element_type_89: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    add_96: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[192]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_44: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  unsqueeze_353 = None
    mul_133: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_357: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_359: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_97: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    add_98: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_97, add_91);  add_97 = None
    convolution_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_98, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_90: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    convert_element_type_91: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    add_99: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[1152]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_45: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  unsqueeze_361 = None
    mul_136: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_365: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_367: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_100: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    relu_30: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    convolution_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_139, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    convert_element_type_92: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    convert_element_type_93: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    add_101: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[1152]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_46: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  unsqueeze_369 = None
    mul_139: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_373: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_375: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_102: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    relu_31: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    convolution_47: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_31, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_94: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    convert_element_type_95: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    add_103: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[192]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_47: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  unsqueeze_377 = None
    mul_142: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_381: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_383: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_104: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    add_105: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_104, add_98);  add_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    convolution_48: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_105, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_96: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    convert_element_type_97: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    add_106: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[1152]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_48: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  unsqueeze_385 = None
    mul_145: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_389: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_391: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_107: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    relu_32: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    convolution_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_148, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    convert_element_type_98: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    convert_element_type_99: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    add_108: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[1152]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_49: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  unsqueeze_393 = None
    mul_148: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_397: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_399: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_109: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    relu_33: "f32[4, 1152, 7, 7]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    convolution_50: "f32[4, 320, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_100: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    convert_element_type_101: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    add_110: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[320]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_50: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  unsqueeze_401 = None
    mul_151: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_405: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_407: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_111: "f32[4, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:159, code: x = self.layers(x)
    convolution_51: "f32[4, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_111, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_102: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_312, torch.float32)
    convert_element_type_103: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_313, torch.float32)
    add_112: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[1280]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_51: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  unsqueeze_409 = None
    mul_154: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_413: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_415: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_113: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    relu_34: "f32[4, 1280, 7, 7]" = torch.ops.aten.relu.default(add_113);  add_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:161, code: x = x.mean([2, 3])
    mean: "f32[4, 1280]" = torch.ops.aten.mean.dim(relu_34, [2, 3])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:162, code: return self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_158, mean, permute);  primals_158 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, mean);  permute_2 = mean = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:161, code: x = x.mean([2, 3])
    unsqueeze_416: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
    unsqueeze_417: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    expand: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_417, [4, 1280, 7, 7]);  unsqueeze_417 = None
    div: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:159, code: x = self.layers(x)
    alias_36: "f32[4, 1280, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_37: "f32[4, 1280, 7, 7]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le: "b8[4, 1280, 7, 7]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[4, 1280, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    add_114: "f32[1280]" = torch.ops.aten.add.Tensor(primals_313, 1e-05);  primals_313 = None
    rsqrt: "f32[1280]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    unsqueeze_418: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_312, 0);  primals_312 = None
    unsqueeze_419: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_52: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_420);  convolution_51 = unsqueeze_420 = None
    mul_156: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_52);  sub_52 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 2, 3]);  mul_156 = None
    mul_161: "f32[1280]" = torch.ops.aten.mul.Tensor(rsqrt, primals_155);  primals_155 = None
    unsqueeze_427: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_161, 0);  mul_161 = None
    unsqueeze_428: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_162: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_429);  where = unsqueeze_429 = None
    mul_163: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_162, add_111, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_162 = add_111 = primals_154 = None
    getitem: "f32[4, 320, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_115: "f32[320]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_1: "f32[320]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    unsqueeze_430: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_431: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_53: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_432);  convolution_50 = unsqueeze_432 = None
    mul_164: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_53);  sub_53 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 2, 3]);  mul_164 = None
    mul_169: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_152);  primals_152 = None
    unsqueeze_439: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_440: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_170: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_441);  getitem = unsqueeze_441 = None
    mul_171: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_170, relu_33, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_170 = primals_151 = None
    getitem_3: "f32[4, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    alias_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_1: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_40, 0);  alias_40 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_3);  le_1 = scalar_tensor_1 = getitem_3 = None
    add_116: "f32[1152]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_2: "f32[1152]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_442: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_443: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    sum_6: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_54: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_444);  convolution_49 = unsqueeze_444 = None
    mul_172: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_54);  sub_54 = None
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 2, 3]);  mul_172 = None
    mul_177: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_149);  primals_149 = None
    unsqueeze_451: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
    unsqueeze_452: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_178: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_453);  where_1 = unsqueeze_453 = None
    mul_179: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_178, relu_32, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_178 = primals_148 = None
    getitem_6: "f32[4, 1152, 7, 7]" = convolution_backward_2[0]
    getitem_7: "f32[1152, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    alias_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_2: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_6);  le_2 = scalar_tensor_2 = getitem_6 = None
    add_117: "f32[1152]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_3: "f32[1152]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    unsqueeze_454: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_455: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_55: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_456);  convolution_48 = unsqueeze_456 = None
    mul_180: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_55);  sub_55 = None
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 2, 3]);  mul_180 = None
    mul_185: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_146);  primals_146 = None
    unsqueeze_463: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_185, 0);  mul_185 = None
    unsqueeze_464: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_186: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_465);  where_2 = unsqueeze_465 = None
    mul_187: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_186, add_105, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_186 = add_105 = primals_145 = None
    getitem_9: "f32[4, 192, 7, 7]" = convolution_backward_3[0]
    getitem_10: "f32[1152, 192, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_118: "f32[192]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_4: "f32[192]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    unsqueeze_466: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_467: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_9, [0, 2, 3])
    sub_56: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_468);  convolution_47 = unsqueeze_468 = None
    mul_188: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_9, sub_56);  sub_56 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None
    mul_193: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_143);  primals_143 = None
    unsqueeze_475: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_193, 0);  mul_193 = None
    unsqueeze_476: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_194: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_477);  unsqueeze_477 = None
    mul_195: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_194, relu_31, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_194 = primals_142 = None
    getitem_12: "f32[4, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[192, 1152, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    alias_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_3: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_12);  le_3 = scalar_tensor_3 = getitem_12 = None
    add_119: "f32[1152]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_5: "f32[1152]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    unsqueeze_478: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_479: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_480);  convolution_46 = unsqueeze_480 = None
    mul_196: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_57);  sub_57 = None
    sum_13: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 2, 3]);  mul_196 = None
    mul_201: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_140);  primals_140 = None
    unsqueeze_487: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
    unsqueeze_488: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_202: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_489);  where_3 = unsqueeze_489 = None
    mul_203: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_202, relu_30, primals_139, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_202 = primals_139 = None
    getitem_15: "f32[4, 1152, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[1152, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    alias_48: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_4: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_15);  le_4 = scalar_tensor_4 = getitem_15 = None
    add_120: "f32[1152]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_6: "f32[1152]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_490: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_491: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_58: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_492);  convolution_45 = unsqueeze_492 = None
    mul_204: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_58);  sub_58 = None
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_209: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_137);  primals_137 = None
    unsqueeze_499: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_500: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_210: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_501);  where_4 = unsqueeze_501 = None
    mul_211: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_210, add_98, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = add_98 = primals_136 = None
    getitem_18: "f32[4, 192, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[1152, 192, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_121: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_9, getitem_18);  getitem_9 = getitem_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_122: "f32[192]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_7: "f32[192]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_502: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_503: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 2, 3])
    sub_59: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_504);  convolution_44 = unsqueeze_504 = None
    mul_212: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, sub_59);  sub_59 = None
    sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
    mul_217: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_134);  primals_134 = None
    unsqueeze_511: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_512: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_218: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, unsqueeze_513);  unsqueeze_513 = None
    mul_219: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_218, relu_29, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_218 = primals_133 = None
    getitem_21: "f32[4, 1152, 7, 7]" = convolution_backward_7[0]
    getitem_22: "f32[192, 1152, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    alias_51: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_52: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_5: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_21);  le_5 = scalar_tensor_5 = getitem_21 = None
    add_123: "f32[1152]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_8: "f32[1152]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_514: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_515: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_516);  convolution_43 = unsqueeze_516 = None
    mul_220: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_60);  sub_60 = None
    sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 2, 3]);  mul_220 = None
    mul_225: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_131);  primals_131 = None
    unsqueeze_523: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_524: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_226: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_525);  where_5 = unsqueeze_525 = None
    mul_227: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_226, relu_28, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_226 = primals_130 = None
    getitem_24: "f32[4, 1152, 7, 7]" = convolution_backward_8[0]
    getitem_25: "f32[1152, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    alias_54: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_55: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_6: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_24);  le_6 = scalar_tensor_6 = getitem_24 = None
    add_124: "f32[1152]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_9: "f32[1152]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_526: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_527: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_528);  convolution_42 = unsqueeze_528 = None
    mul_228: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_61);  sub_61 = None
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 2, 3]);  mul_228 = None
    mul_233: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_128);  primals_128 = None
    unsqueeze_535: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_233, 0);  mul_233 = None
    unsqueeze_536: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_234: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_537);  where_6 = unsqueeze_537 = None
    mul_235: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_234, add_91, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_234 = add_91 = primals_127 = None
    getitem_27: "f32[4, 192, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[1152, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_125: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_121, getitem_27);  add_121 = getitem_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_126: "f32[192]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_10: "f32[192]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    unsqueeze_538: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_539: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 2, 3])
    sub_62: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_540);  convolution_41 = unsqueeze_540 = None
    mul_236: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_125, sub_62);  sub_62 = None
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 2, 3]);  mul_236 = None
    mul_241: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_125);  primals_125 = None
    unsqueeze_547: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_241, 0);  mul_241 = None
    unsqueeze_548: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_242: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_125, unsqueeze_549);  unsqueeze_549 = None
    mul_243: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_242, relu_27, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_242 = primals_124 = None
    getitem_30: "f32[4, 1152, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[192, 1152, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    alias_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_58: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_7: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_30);  le_7 = scalar_tensor_7 = getitem_30 = None
    add_127: "f32[1152]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_11: "f32[1152]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    unsqueeze_550: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_551: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_63: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_552);  convolution_40 = unsqueeze_552 = None
    mul_244: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_63);  sub_63 = None
    sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 2, 3]);  mul_244 = None
    mul_249: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_122);  primals_122 = None
    unsqueeze_559: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
    unsqueeze_560: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_250: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_561);  where_7 = unsqueeze_561 = None
    mul_251: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_250, relu_26, primals_121, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_250 = primals_121 = None
    getitem_33: "f32[4, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_34: "f32[1152, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    alias_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_8: "b8[4, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[4, 1152, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_33);  le_8 = scalar_tensor_8 = getitem_33 = None
    add_128: "f32[1152]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_12: "f32[1152]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_562: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_563: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    sum_26: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_64: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_564);  convolution_39 = unsqueeze_564 = None
    mul_252: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_64);  sub_64 = None
    sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_257: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_119);  primals_119 = None
    unsqueeze_571: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_572: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_258: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_573);  where_8 = unsqueeze_573 = None
    mul_259: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_258, add_84, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_258 = add_84 = primals_118 = None
    getitem_36: "f32[4, 192, 7, 7]" = convolution_backward_12[0]
    getitem_37: "f32[1152, 192, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_129: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_125, getitem_36);  add_125 = getitem_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_130: "f32[192]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_13: "f32[192]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_574: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_575: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3])
    sub_65: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_576);  convolution_38 = unsqueeze_576 = None
    mul_260: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_129, sub_65);  sub_65 = None
    sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 2, 3]);  mul_260 = None
    mul_265: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_116);  primals_116 = None
    unsqueeze_583: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_584: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_266: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_129, unsqueeze_585);  add_129 = unsqueeze_585 = None
    mul_267: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_266, relu_25, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = primals_115 = None
    getitem_39: "f32[4, 576, 7, 7]" = convolution_backward_13[0]
    getitem_40: "f32[192, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    alias_63: "f32[4, 576, 7, 7]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_64: "f32[4, 576, 7, 7]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_9: "b8[4, 576, 7, 7]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 576, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_39);  le_9 = scalar_tensor_9 = getitem_39 = None
    add_131: "f32[576]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_14: "f32[576]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_586: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_587: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_66: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_588);  convolution_37 = unsqueeze_588 = None
    mul_268: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_66);  sub_66 = None
    sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 2, 3]);  mul_268 = None
    mul_273: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_113);  primals_113 = None
    unsqueeze_595: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
    unsqueeze_596: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_274: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_597);  where_9 = unsqueeze_597 = None
    mul_275: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_274, relu_24, primals_112, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_274 = primals_112 = None
    getitem_42: "f32[4, 576, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[576, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    alias_66: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_67: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_10: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_42);  le_10 = scalar_tensor_10 = getitem_42 = None
    add_132: "f32[576]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_15: "f32[576]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_598: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_599: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_67: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_600);  convolution_36 = unsqueeze_600 = None
    mul_276: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_67);  sub_67 = None
    sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3]);  mul_276 = None
    mul_281: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_110);  primals_110 = None
    unsqueeze_607: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_608: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_282: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_609);  where_10 = unsqueeze_609 = None
    mul_283: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_282, add_78, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_282 = add_78 = primals_109 = None
    getitem_45: "f32[4, 96, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_133: "f32[96]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_16: "f32[96]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_610: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_611: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_45, [0, 2, 3])
    sub_68: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_612);  convolution_35 = unsqueeze_612 = None
    mul_284: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, sub_68);  sub_68 = None
    sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
    mul_289: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_107);  primals_107 = None
    unsqueeze_619: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
    unsqueeze_620: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_290: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_621);  unsqueeze_621 = None
    mul_291: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_290, relu_23, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_290 = primals_106 = None
    getitem_48: "f32[4, 576, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[96, 576, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    alias_69: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_70: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_11: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_48);  le_11 = scalar_tensor_11 = getitem_48 = None
    add_134: "f32[576]" = torch.ops.aten.add.Tensor(primals_262, 1e-05);  primals_262 = None
    rsqrt_17: "f32[576]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_622: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
    unsqueeze_623: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    sum_36: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_69: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_624);  convolution_34 = unsqueeze_624 = None
    mul_292: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_69);  sub_69 = None
    sum_37: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 2, 3]);  mul_292 = None
    mul_297: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_104);  primals_104 = None
    unsqueeze_631: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
    unsqueeze_632: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_298: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_633);  where_11 = unsqueeze_633 = None
    mul_299: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_298, relu_22, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_298 = primals_103 = None
    getitem_51: "f32[4, 576, 14, 14]" = convolution_backward_17[0]
    getitem_52: "f32[576, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    alias_72: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_73: "f32[4, 576, 14, 14]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_12: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, getitem_51);  le_12 = scalar_tensor_12 = getitem_51 = None
    add_135: "f32[576]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_18: "f32[576]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    unsqueeze_634: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_635: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    sum_38: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_70: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_636);  convolution_33 = unsqueeze_636 = None
    mul_300: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_70);  sub_70 = None
    sum_39: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
    mul_305: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_101);  primals_101 = None
    unsqueeze_643: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
    unsqueeze_644: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_306: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_645);  where_12 = unsqueeze_645 = None
    mul_307: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_306, add_71, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_306 = add_71 = primals_100 = None
    getitem_54: "f32[4, 96, 14, 14]" = convolution_backward_18[0]
    getitem_55: "f32[576, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_136: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_45, getitem_54);  getitem_45 = getitem_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_137: "f32[96]" = torch.ops.aten.add.Tensor(primals_256, 1e-05);  primals_256 = None
    rsqrt_19: "f32[96]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_646: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_255, 0);  primals_255 = None
    unsqueeze_647: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 2, 3])
    sub_71: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_648);  convolution_32 = unsqueeze_648 = None
    mul_308: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, sub_71);  sub_71 = None
    sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 2, 3]);  mul_308 = None
    mul_313: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_98);  primals_98 = None
    unsqueeze_655: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
    unsqueeze_656: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_314: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, unsqueeze_657);  add_136 = unsqueeze_657 = None
    mul_315: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_314, relu_21, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_314 = primals_97 = None
    getitem_57: "f32[4, 480, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[96, 480, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    alias_75: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_76: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_13: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_57);  le_13 = scalar_tensor_13 = getitem_57 = None
    add_138: "f32[480]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_20: "f32[480]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_658: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_659: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_72: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_660);  convolution_31 = unsqueeze_660 = None
    mul_316: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_72);  sub_72 = None
    sum_43: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 2, 3]);  mul_316 = None
    mul_321: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_95);  primals_95 = None
    unsqueeze_667: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
    unsqueeze_668: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_322: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_669);  where_13 = unsqueeze_669 = None
    mul_323: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_322, relu_20, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_322 = primals_94 = None
    getitem_60: "f32[4, 480, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[480, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    alias_78: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_79: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_14: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_60);  le_14 = scalar_tensor_14 = getitem_60 = None
    add_139: "f32[480]" = torch.ops.aten.add.Tensor(primals_250, 1e-05);  primals_250 = None
    rsqrt_21: "f32[480]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_670: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_249, 0);  primals_249 = None
    unsqueeze_671: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_73: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_672);  convolution_30 = unsqueeze_672 = None
    mul_324: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_73);  sub_73 = None
    sum_45: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 2, 3]);  mul_324 = None
    mul_329: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_92);  primals_92 = None
    unsqueeze_679: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
    unsqueeze_680: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_330: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_681);  where_14 = unsqueeze_681 = None
    mul_331: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_330, add_65, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_330 = add_65 = primals_91 = None
    getitem_63: "f32[4, 80, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[480, 80, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_140: "f32[80]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_22: "f32[80]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_682: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_683: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    sum_46: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_63, [0, 2, 3])
    sub_74: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_684);  convolution_29 = unsqueeze_684 = None
    mul_332: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, sub_74);  sub_74 = None
    sum_47: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 2, 3]);  mul_332 = None
    mul_337: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_89);  primals_89 = None
    unsqueeze_691: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
    unsqueeze_692: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_338: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, unsqueeze_693);  unsqueeze_693 = None
    mul_339: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_338, relu_19, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_338 = primals_88 = None
    getitem_66: "f32[4, 480, 14, 14]" = convolution_backward_22[0]
    getitem_67: "f32[80, 480, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    alias_81: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_82: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_15: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_66);  le_15 = scalar_tensor_15 = getitem_66 = None
    add_141: "f32[480]" = torch.ops.aten.add.Tensor(primals_244, 1e-05);  primals_244 = None
    rsqrt_23: "f32[480]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_694: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_243, 0);  primals_243 = None
    unsqueeze_695: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_75: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_696);  convolution_28 = unsqueeze_696 = None
    mul_340: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_75);  sub_75 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 2, 3]);  mul_340 = None
    mul_345: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_86);  primals_86 = None
    unsqueeze_703: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_704: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_346: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_705);  where_15 = unsqueeze_705 = None
    mul_347: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_346, relu_18, primals_85, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_346 = primals_85 = None
    getitem_69: "f32[4, 480, 14, 14]" = convolution_backward_23[0]
    getitem_70: "f32[480, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    alias_84: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_85: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_16: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_69);  le_16 = scalar_tensor_16 = getitem_69 = None
    add_142: "f32[480]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_24: "f32[480]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    unsqueeze_706: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_707: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    sum_50: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_76: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_708);  convolution_27 = unsqueeze_708 = None
    mul_348: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_76);  sub_76 = None
    sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_353: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_83);  primals_83 = None
    unsqueeze_715: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_716: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_354: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_717);  where_16 = unsqueeze_717 = None
    mul_355: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_354, add_58, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = add_58 = primals_82 = None
    getitem_72: "f32[4, 80, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[480, 80, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_143: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_63, getitem_72);  getitem_63 = getitem_72 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_144: "f32[80]" = torch.ops.aten.add.Tensor(primals_238, 1e-05);  primals_238 = None
    rsqrt_25: "f32[80]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_718: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_237, 0);  primals_237 = None
    unsqueeze_719: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    sum_52: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3])
    sub_77: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_720);  convolution_26 = unsqueeze_720 = None
    mul_356: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, sub_77);  sub_77 = None
    sum_53: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 2, 3]);  mul_356 = None
    mul_361: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_80);  primals_80 = None
    unsqueeze_727: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
    unsqueeze_728: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_362: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, unsqueeze_729);  unsqueeze_729 = None
    mul_363: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_362, relu_17, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_362 = primals_79 = None
    getitem_75: "f32[4, 480, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[80, 480, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    alias_87: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_88: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_17: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_75);  le_17 = scalar_tensor_17 = getitem_75 = None
    add_145: "f32[480]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_26: "f32[480]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_730: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_731: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    sum_54: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_78: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_732);  convolution_25 = unsqueeze_732 = None
    mul_364: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_78);  sub_78 = None
    sum_55: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_369: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_77);  primals_77 = None
    unsqueeze_739: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_740: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_370: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_741);  where_17 = unsqueeze_741 = None
    mul_371: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_370, relu_16, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_370 = primals_76 = None
    getitem_78: "f32[4, 480, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[480, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    alias_90: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_91: "f32[4, 480, 14, 14]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_18: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, getitem_78);  le_18 = scalar_tensor_18 = getitem_78 = None
    add_146: "f32[480]" = torch.ops.aten.add.Tensor(primals_232, 1e-05);  primals_232 = None
    rsqrt_27: "f32[480]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_742: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_231, 0);  primals_231 = None
    unsqueeze_743: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_79: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_744);  convolution_24 = unsqueeze_744 = None
    mul_372: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_79);  sub_79 = None
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 2, 3]);  mul_372 = None
    mul_377: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_74);  primals_74 = None
    unsqueeze_751: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_752: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_378: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_753);  where_18 = unsqueeze_753 = None
    mul_379: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_378, add_51, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_378 = add_51 = primals_73 = None
    getitem_81: "f32[4, 80, 14, 14]" = convolution_backward_27[0]
    getitem_82: "f32[480, 80, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_147: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_143, getitem_81);  add_143 = getitem_81 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_148: "f32[80]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_28: "f32[80]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_754: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_755: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    sum_58: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
    sub_80: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_756);  convolution_23 = unsqueeze_756 = None
    mul_380: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_147, sub_80);  sub_80 = None
    sum_59: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3]);  mul_380 = None
    mul_385: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_71);  primals_71 = None
    unsqueeze_763: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
    unsqueeze_764: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_386: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_147, unsqueeze_765);  add_147 = unsqueeze_765 = None
    mul_387: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_386, relu_15, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_386 = primals_70 = None
    getitem_84: "f32[4, 240, 14, 14]" = convolution_backward_28[0]
    getitem_85: "f32[80, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    alias_93: "f32[4, 240, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_94: "f32[4, 240, 14, 14]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_19: "b8[4, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[4, 240, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_84);  le_19 = scalar_tensor_19 = getitem_84 = None
    add_149: "f32[240]" = torch.ops.aten.add.Tensor(primals_226, 1e-05);  primals_226 = None
    rsqrt_29: "f32[240]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_766: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_225, 0);  primals_225 = None
    unsqueeze_767: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    sum_60: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_81: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_768);  convolution_22 = unsqueeze_768 = None
    mul_388: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_81);  sub_81 = None
    sum_61: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3]);  mul_388 = None
    mul_393: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_68);  primals_68 = None
    unsqueeze_775: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
    unsqueeze_776: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_394: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_777);  where_19 = unsqueeze_777 = None
    mul_395: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_394, relu_14, primals_67, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_394 = primals_67 = None
    getitem_87: "f32[4, 240, 28, 28]" = convolution_backward_29[0]
    getitem_88: "f32[240, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    alias_96: "f32[4, 240, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_97: "f32[4, 240, 28, 28]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_20: "b8[4, 240, 28, 28]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[4, 240, 28, 28]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_87);  le_20 = scalar_tensor_20 = getitem_87 = None
    add_150: "f32[240]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_30: "f32[240]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    unsqueeze_778: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_779: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_82: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_780);  convolution_21 = unsqueeze_780 = None
    mul_396: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_82);  sub_82 = None
    sum_63: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
    mul_401: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_65);  primals_65 = None
    unsqueeze_787: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_788: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_402: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_789);  where_20 = unsqueeze_789 = None
    mul_403: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_402, add_45, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_402 = add_45 = primals_64 = None
    getitem_90: "f32[4, 40, 28, 28]" = convolution_backward_30[0]
    getitem_91: "f32[240, 40, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_151: "f32[40]" = torch.ops.aten.add.Tensor(primals_220, 1e-05);  primals_220 = None
    rsqrt_31: "f32[40]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_790: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_219, 0);  primals_219 = None
    unsqueeze_791: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    sum_64: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_90, [0, 2, 3])
    sub_83: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_792);  convolution_20 = unsqueeze_792 = None
    mul_404: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_90, sub_83);  sub_83 = None
    sum_65: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3]);  mul_404 = None
    mul_409: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_62);  primals_62 = None
    unsqueeze_799: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_800: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_410: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_90, unsqueeze_801);  unsqueeze_801 = None
    mul_411: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_410, relu_13, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_410 = primals_61 = None
    getitem_93: "f32[4, 120, 28, 28]" = convolution_backward_31[0]
    getitem_94: "f32[40, 120, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    alias_99: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_100: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_21: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_93);  le_21 = scalar_tensor_21 = getitem_93 = None
    add_152: "f32[120]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_32: "f32[120]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_802: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_803: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    sum_66: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_84: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_804);  convolution_19 = unsqueeze_804 = None
    mul_412: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_84);  sub_84 = None
    sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 2, 3]);  mul_412 = None
    mul_417: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_59);  primals_59 = None
    unsqueeze_811: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_812: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_418: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_813);  where_21 = unsqueeze_813 = None
    mul_419: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_418, relu_12, primals_58, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_418 = primals_58 = None
    getitem_96: "f32[4, 120, 28, 28]" = convolution_backward_32[0]
    getitem_97: "f32[120, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
    alias_102: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_103: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_22: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_96);  le_22 = scalar_tensor_22 = getitem_96 = None
    add_153: "f32[120]" = torch.ops.aten.add.Tensor(primals_214, 1e-05);  primals_214 = None
    rsqrt_33: "f32[120]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_814: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_213, 0);  primals_213 = None
    unsqueeze_815: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_85: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_816);  convolution_18 = unsqueeze_816 = None
    mul_420: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_85);  sub_85 = None
    sum_69: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3]);  mul_420 = None
    mul_425: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_56);  primals_56 = None
    unsqueeze_823: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_824: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_426: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_825);  where_22 = unsqueeze_825 = None
    mul_427: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_426, add_38, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_426 = add_38 = primals_55 = None
    getitem_99: "f32[4, 40, 28, 28]" = convolution_backward_33[0]
    getitem_100: "f32[120, 40, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_154: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_90, getitem_99);  getitem_90 = getitem_99 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_155: "f32[40]" = torch.ops.aten.add.Tensor(primals_211, 1e-05);  primals_211 = None
    rsqrt_34: "f32[40]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_826: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_210, 0);  primals_210 = None
    unsqueeze_827: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    sum_70: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 2, 3])
    sub_86: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_828);  convolution_17 = unsqueeze_828 = None
    mul_428: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_154, sub_86);  sub_86 = None
    sum_71: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 2, 3]);  mul_428 = None
    mul_433: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_53);  primals_53 = None
    unsqueeze_835: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
    unsqueeze_836: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_434: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_154, unsqueeze_837);  unsqueeze_837 = None
    mul_435: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_434, relu_11, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_52 = None
    getitem_102: "f32[4, 120, 28, 28]" = convolution_backward_34[0]
    getitem_103: "f32[40, 120, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    alias_105: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_106: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_23: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_102);  le_23 = scalar_tensor_23 = getitem_102 = None
    add_156: "f32[120]" = torch.ops.aten.add.Tensor(primals_208, 1e-05);  primals_208 = None
    rsqrt_35: "f32[120]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_838: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_207, 0);  primals_207 = None
    unsqueeze_839: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    sum_72: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_87: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_840);  convolution_16 = unsqueeze_840 = None
    mul_436: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_87);  sub_87 = None
    sum_73: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_441: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_50);  primals_50 = None
    unsqueeze_847: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_848: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_442: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_849);  where_23 = unsqueeze_849 = None
    mul_443: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_442, relu_10, primals_49, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_442 = primals_49 = None
    getitem_105: "f32[4, 120, 28, 28]" = convolution_backward_35[0]
    getitem_106: "f32[120, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
    alias_108: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_109: "f32[4, 120, 28, 28]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_24: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, getitem_105);  le_24 = scalar_tensor_24 = getitem_105 = None
    add_157: "f32[120]" = torch.ops.aten.add.Tensor(primals_205, 1e-05);  primals_205 = None
    rsqrt_36: "f32[120]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_850: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_204, 0);  primals_204 = None
    unsqueeze_851: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_88: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_852);  convolution_15 = unsqueeze_852 = None
    mul_444: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_88);  sub_88 = None
    sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3]);  mul_444 = None
    mul_449: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_47);  primals_47 = None
    unsqueeze_859: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_860: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_450: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_861);  where_24 = unsqueeze_861 = None
    mul_451: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_450, add_31, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_450 = add_31 = primals_46 = None
    getitem_108: "f32[4, 40, 28, 28]" = convolution_backward_36[0]
    getitem_109: "f32[120, 40, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_158: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_154, getitem_108);  add_154 = getitem_108 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_159: "f32[40]" = torch.ops.aten.add.Tensor(primals_202, 1e-05);  primals_202 = None
    rsqrt_37: "f32[40]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_862: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_201, 0);  primals_201 = None
    unsqueeze_863: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    sum_76: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 2, 3])
    sub_89: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_864);  convolution_14 = unsqueeze_864 = None
    mul_452: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_158, sub_89);  sub_89 = None
    sum_77: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 2, 3]);  mul_452 = None
    mul_457: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_44);  primals_44 = None
    unsqueeze_871: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
    unsqueeze_872: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_458: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_158, unsqueeze_873);  add_158 = unsqueeze_873 = None
    mul_459: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_458, relu_9, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_458 = primals_43 = None
    getitem_111: "f32[4, 72, 28, 28]" = convolution_backward_37[0]
    getitem_112: "f32[40, 72, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    alias_111: "f32[4, 72, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_112: "f32[4, 72, 28, 28]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_25: "b8[4, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[4, 72, 28, 28]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_111);  le_25 = scalar_tensor_25 = getitem_111 = None
    add_160: "f32[72]" = torch.ops.aten.add.Tensor(primals_199, 1e-05);  primals_199 = None
    rsqrt_38: "f32[72]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    unsqueeze_874: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_198, 0);  primals_198 = None
    unsqueeze_875: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    sum_78: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_90: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_876);  convolution_13 = unsqueeze_876 = None
    mul_460: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_90);  sub_90 = None
    sum_79: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 2, 3]);  mul_460 = None
    mul_465: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_41);  primals_41 = None
    unsqueeze_883: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_884: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_466: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_885);  where_25 = unsqueeze_885 = None
    mul_467: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_466, relu_8, primals_40, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_466 = primals_40 = None
    getitem_114: "f32[4, 72, 56, 56]" = convolution_backward_38[0]
    getitem_115: "f32[72, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    alias_114: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_115: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_26: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_114);  le_26 = scalar_tensor_26 = getitem_114 = None
    add_161: "f32[72]" = torch.ops.aten.add.Tensor(primals_196, 1e-05);  primals_196 = None
    rsqrt_39: "f32[72]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_886: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_195, 0);  primals_195 = None
    unsqueeze_887: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    sum_80: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_91: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_888);  convolution_12 = unsqueeze_888 = None
    mul_468: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, sub_91);  sub_91 = None
    sum_81: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 2, 3]);  mul_468 = None
    mul_473: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_38);  primals_38 = None
    unsqueeze_895: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_896: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_474: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_897);  where_26 = unsqueeze_897 = None
    mul_475: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_474, add_25, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = add_25 = primals_37 = None
    getitem_117: "f32[4, 24, 56, 56]" = convolution_backward_39[0]
    getitem_118: "f32[72, 24, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_162: "f32[24]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_40: "f32[24]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_898: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_899: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    sum_82: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_117, [0, 2, 3])
    sub_92: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_900);  convolution_11 = unsqueeze_900 = None
    mul_476: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_117, sub_92);  sub_92 = None
    sum_83: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_481: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_35);  primals_35 = None
    unsqueeze_907: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_908: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_482: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_117, unsqueeze_909);  unsqueeze_909 = None
    mul_483: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_482, relu_7, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_482 = primals_34 = None
    getitem_120: "f32[4, 72, 56, 56]" = convolution_backward_40[0]
    getitem_121: "f32[24, 72, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    alias_117: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_118: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_27: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_120);  le_27 = scalar_tensor_27 = getitem_120 = None
    add_163: "f32[72]" = torch.ops.aten.add.Tensor(primals_190, 1e-05);  primals_190 = None
    rsqrt_41: "f32[72]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    unsqueeze_910: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_189, 0);  primals_189 = None
    unsqueeze_911: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    sum_84: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_93: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_912);  convolution_10 = unsqueeze_912 = None
    mul_484: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, sub_93);  sub_93 = None
    sum_85: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2, 3]);  mul_484 = None
    mul_489: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_32);  primals_32 = None
    unsqueeze_919: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
    unsqueeze_920: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_490: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_921);  where_27 = unsqueeze_921 = None
    mul_491: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_490, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_490 = primals_31 = None
    getitem_123: "f32[4, 72, 56, 56]" = convolution_backward_41[0]
    getitem_124: "f32[72, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    alias_120: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_121: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_28: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_123);  le_28 = scalar_tensor_28 = getitem_123 = None
    add_164: "f32[72]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_42: "f32[72]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_922: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_923: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    sum_86: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_94: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_924);  convolution_9 = unsqueeze_924 = None
    mul_492: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_94);  sub_94 = None
    sum_87: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 2, 3]);  mul_492 = None
    mul_497: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_29);  primals_29 = None
    unsqueeze_931: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
    unsqueeze_932: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_498: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_933);  where_28 = unsqueeze_933 = None
    mul_499: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_498, add_18, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = add_18 = primals_28 = None
    getitem_126: "f32[4, 24, 56, 56]" = convolution_backward_42[0]
    getitem_127: "f32[72, 24, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_165: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_117, getitem_126);  getitem_117 = getitem_126 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_166: "f32[24]" = torch.ops.aten.add.Tensor(primals_184, 1e-05);  primals_184 = None
    rsqrt_43: "f32[24]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_934: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_183, 0);  primals_183 = None
    unsqueeze_935: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    sum_88: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 2, 3])
    sub_95: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_936);  convolution_8 = unsqueeze_936 = None
    mul_500: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_165, sub_95);  sub_95 = None
    sum_89: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_505: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_26);  primals_26 = None
    unsqueeze_943: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_944: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_506: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_165, unsqueeze_945);  unsqueeze_945 = None
    mul_507: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_506, relu_5, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = primals_25 = None
    getitem_129: "f32[4, 72, 56, 56]" = convolution_backward_43[0]
    getitem_130: "f32[24, 72, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    alias_123: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_124: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_29: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_129);  le_29 = scalar_tensor_29 = getitem_129 = None
    add_167: "f32[72]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_44: "f32[72]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_946: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_947: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    sum_90: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_96: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_948);  convolution_7 = unsqueeze_948 = None
    mul_508: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_96);  sub_96 = None
    sum_91: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
    mul_513: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_23);  primals_23 = None
    unsqueeze_955: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_956: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_514: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_957);  where_29 = unsqueeze_957 = None
    mul_515: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_514, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_514 = primals_22 = None
    getitem_132: "f32[4, 72, 56, 56]" = convolution_backward_44[0]
    getitem_133: "f32[72, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    alias_126: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_127: "f32[4, 72, 56, 56]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_30: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_132);  le_30 = scalar_tensor_30 = getitem_132 = None
    add_168: "f32[72]" = torch.ops.aten.add.Tensor(primals_178, 1e-05);  primals_178 = None
    rsqrt_45: "f32[72]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    unsqueeze_958: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_177, 0);  primals_177 = None
    unsqueeze_959: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    sum_92: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_97: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_960);  convolution_6 = unsqueeze_960 = None
    mul_516: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_97);  sub_97 = None
    sum_93: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_521: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_20);  primals_20 = None
    unsqueeze_967: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_968: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_522: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_969);  where_30 = unsqueeze_969 = None
    mul_523: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_522, add_11, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_522 = add_11 = primals_19 = None
    getitem_135: "f32[4, 24, 56, 56]" = convolution_backward_45[0]
    getitem_136: "f32[72, 24, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:61, code: return self.layers(input) + input
    add_169: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_165, getitem_135);  add_165 = getitem_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:63, code: return self.layers(input)
    add_170: "f32[24]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_46: "f32[24]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_970: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_971: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 2, 3])
    sub_98: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_972);  convolution_5 = unsqueeze_972 = None
    mul_524: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_169, sub_98);  sub_98 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
    mul_529: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_17);  primals_17 = None
    unsqueeze_979: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
    unsqueeze_980: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_530: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_169, unsqueeze_981);  add_169 = unsqueeze_981 = None
    mul_531: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_530, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_530 = primals_16 = None
    getitem_138: "f32[4, 48, 56, 56]" = convolution_backward_46[0]
    getitem_139: "f32[24, 48, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    alias_129: "f32[4, 48, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_130: "f32[4, 48, 56, 56]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_31: "b8[4, 48, 56, 56]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[4, 48, 56, 56]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_138);  le_31 = scalar_tensor_31 = getitem_138 = None
    add_171: "f32[48]" = torch.ops.aten.add.Tensor(primals_172, 1e-05);  primals_172 = None
    rsqrt_47: "f32[48]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_982: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(primals_171, 0);  primals_171 = None
    unsqueeze_983: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    sum_96: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_99: "f32[4, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_984);  convolution_4 = unsqueeze_984 = None
    mul_532: "f32[4, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_99);  sub_99 = None
    sum_97: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 2, 3]);  mul_532 = None
    mul_537: "f32[48]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_14);  primals_14 = None
    unsqueeze_991: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_992: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_538: "f32[4, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_993);  where_31 = unsqueeze_993 = None
    mul_539: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_538, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_538 = primals_13 = None
    getitem_141: "f32[4, 48, 112, 112]" = convolution_backward_47[0]
    getitem_142: "f32[48, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    alias_132: "f32[4, 48, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_133: "f32[4, 48, 112, 112]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_32: "b8[4, 48, 112, 112]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[4, 48, 112, 112]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_141);  le_32 = scalar_tensor_32 = getitem_141 = None
    add_172: "f32[48]" = torch.ops.aten.add.Tensor(primals_169, 1e-05);  primals_169 = None
    rsqrt_48: "f32[48]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_994: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(primals_168, 0);  primals_168 = None
    unsqueeze_995: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    sum_98: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_100: "f32[4, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_996);  convolution_3 = unsqueeze_996 = None
    mul_540: "f32[4, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, sub_100);  sub_100 = None
    sum_99: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
    mul_545: "f32[48]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_11);  primals_11 = None
    unsqueeze_1003: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_1004: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_546: "f32[4, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_1005);  where_32 = unsqueeze_1005 = None
    mul_547: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_546, add_5, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_546 = add_5 = primals_10 = None
    getitem_144: "f32[4, 16, 112, 112]" = convolution_backward_48[0]
    getitem_145: "f32[48, 16, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mnasnet.py:159, code: x = self.layers(x)
    add_173: "f32[16]" = torch.ops.aten.add.Tensor(primals_166, 1e-05);  primals_166 = None
    rsqrt_49: "f32[16]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_1006: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_165, 0);  primals_165 = None
    unsqueeze_1007: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    sum_100: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_144, [0, 2, 3])
    sub_101: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1008);  convolution_2 = unsqueeze_1008 = None
    mul_548: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, sub_101);  sub_101 = None
    sum_101: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_553: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_8);  primals_8 = None
    unsqueeze_1015: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_1016: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_554: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_144, unsqueeze_1017);  getitem_144 = unsqueeze_1017 = None
    mul_555: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_554, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = primals_7 = None
    getitem_147: "f32[4, 32, 112, 112]" = convolution_backward_49[0]
    getitem_148: "f32[16, 32, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    alias_135: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_136: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_33: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_147);  le_33 = scalar_tensor_33 = getitem_147 = None
    add_174: "f32[32]" = torch.ops.aten.add.Tensor(primals_163, 1e-05);  primals_163 = None
    rsqrt_50: "f32[32]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_1018: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_162, 0);  primals_162 = None
    unsqueeze_1019: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_102: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1020);  convolution_1 = unsqueeze_1020 = None
    mul_556: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, sub_102);  sub_102 = None
    sum_103: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None
    mul_561: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_5);  primals_5 = None
    unsqueeze_1027: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_1028: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_562: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_1029);  where_33 = unsqueeze_1029 = None
    mul_563: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_562, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_562 = primals_4 = None
    getitem_150: "f32[4, 32, 112, 112]" = convolution_backward_50[0]
    getitem_151: "f32[32, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    alias_138: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_139: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_34: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_150);  le_34 = scalar_tensor_34 = getitem_150 = None
    add_175: "f32[32]" = torch.ops.aten.add.Tensor(primals_160, 1e-05);  primals_160 = None
    rsqrt_51: "f32[32]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_1030: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_159, 0);  primals_159 = None
    unsqueeze_1031: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
    unsqueeze_1032: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
    sum_104: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_103: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1032);  convolution = unsqueeze_1032 = None
    mul_564: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, sub_103);  sub_103 = None
    sum_105: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3]);  mul_564 = None
    mul_569: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_2);  primals_2 = None
    unsqueeze_1039: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_1040: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
    unsqueeze_1041: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
    mul_570: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_1041);  where_34 = unsqueeze_1041 = None
    mul_571: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_570, primals_315, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_570 = primals_315 = primals_1 = None
    getitem_154: "f32[32, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    return pytree.tree_unflatten([addmm, getitem_154, mul_571, sum_104, getitem_151, mul_563, sum_102, getitem_148, mul_555, sum_100, getitem_145, mul_547, sum_98, getitem_142, mul_539, sum_96, getitem_139, mul_531, sum_94, getitem_136, mul_523, sum_92, getitem_133, mul_515, sum_90, getitem_130, mul_507, sum_88, getitem_127, mul_499, sum_86, getitem_124, mul_491, sum_84, getitem_121, mul_483, sum_82, getitem_118, mul_475, sum_80, getitem_115, mul_467, sum_78, getitem_112, mul_459, sum_76, getitem_109, mul_451, sum_74, getitem_106, mul_443, sum_72, getitem_103, mul_435, sum_70, getitem_100, mul_427, sum_68, getitem_97, mul_419, sum_66, getitem_94, mul_411, sum_64, getitem_91, mul_403, sum_62, getitem_88, mul_395, sum_60, getitem_85, mul_387, sum_58, getitem_82, mul_379, sum_56, getitem_79, mul_371, sum_54, getitem_76, mul_363, sum_52, getitem_73, mul_355, sum_50, getitem_70, mul_347, sum_48, getitem_67, mul_339, sum_46, getitem_64, mul_331, sum_44, getitem_61, mul_323, sum_42, getitem_58, mul_315, sum_40, getitem_55, mul_307, sum_38, getitem_52, mul_299, sum_36, getitem_49, mul_291, sum_34, getitem_46, mul_283, sum_32, getitem_43, mul_275, sum_30, getitem_40, mul_267, sum_28, getitem_37, mul_259, sum_26, getitem_34, mul_251, sum_24, getitem_31, mul_243, sum_22, getitem_28, mul_235, sum_20, getitem_25, mul_227, sum_18, getitem_22, mul_219, sum_16, getitem_19, mul_211, sum_14, getitem_16, mul_203, sum_12, getitem_13, mul_195, sum_10, getitem_10, mul_187, sum_8, getitem_7, mul_179, sum_6, getitem_4, mul_171, sum_4, getitem_1, mul_163, sum_2, permute_4, view, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    