from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[24, 3, 3, 3]"; primals_2: "f32[24]"; primals_3: "f32[24]"; primals_4: "f32[24, 1, 3, 3]"; primals_5: "f32[24]"; primals_6: "f32[24]"; primals_7: "f32[58, 24, 1, 1]"; primals_8: "f32[58]"; primals_9: "f32[58]"; primals_10: "f32[58, 24, 1, 1]"; primals_11: "f32[58]"; primals_12: "f32[58]"; primals_13: "f32[58, 1, 3, 3]"; primals_14: "f32[58]"; primals_15: "f32[58]"; primals_16: "f32[58, 58, 1, 1]"; primals_17: "f32[58]"; primals_18: "f32[58]"; primals_19: "f32[58, 58, 1, 1]"; primals_20: "f32[58]"; primals_21: "f32[58]"; primals_22: "f32[58, 1, 3, 3]"; primals_23: "f32[58]"; primals_24: "f32[58]"; primals_25: "f32[58, 58, 1, 1]"; primals_26: "f32[58]"; primals_27: "f32[58]"; primals_28: "f32[58, 58, 1, 1]"; primals_29: "f32[58]"; primals_30: "f32[58]"; primals_31: "f32[58, 1, 3, 3]"; primals_32: "f32[58]"; primals_33: "f32[58]"; primals_34: "f32[58, 58, 1, 1]"; primals_35: "f32[58]"; primals_36: "f32[58]"; primals_37: "f32[58, 58, 1, 1]"; primals_38: "f32[58]"; primals_39: "f32[58]"; primals_40: "f32[58, 1, 3, 3]"; primals_41: "f32[58]"; primals_42: "f32[58]"; primals_43: "f32[58, 58, 1, 1]"; primals_44: "f32[58]"; primals_45: "f32[58]"; primals_46: "f32[116, 1, 3, 3]"; primals_47: "f32[116]"; primals_48: "f32[116]"; primals_49: "f32[116, 116, 1, 1]"; primals_50: "f32[116]"; primals_51: "f32[116]"; primals_52: "f32[116, 116, 1, 1]"; primals_53: "f32[116]"; primals_54: "f32[116]"; primals_55: "f32[116, 1, 3, 3]"; primals_56: "f32[116]"; primals_57: "f32[116]"; primals_58: "f32[116, 116, 1, 1]"; primals_59: "f32[116]"; primals_60: "f32[116]"; primals_61: "f32[116, 116, 1, 1]"; primals_62: "f32[116]"; primals_63: "f32[116]"; primals_64: "f32[116, 1, 3, 3]"; primals_65: "f32[116]"; primals_66: "f32[116]"; primals_67: "f32[116, 116, 1, 1]"; primals_68: "f32[116]"; primals_69: "f32[116]"; primals_70: "f32[116, 116, 1, 1]"; primals_71: "f32[116]"; primals_72: "f32[116]"; primals_73: "f32[116, 1, 3, 3]"; primals_74: "f32[116]"; primals_75: "f32[116]"; primals_76: "f32[116, 116, 1, 1]"; primals_77: "f32[116]"; primals_78: "f32[116]"; primals_79: "f32[116, 116, 1, 1]"; primals_80: "f32[116]"; primals_81: "f32[116]"; primals_82: "f32[116, 1, 3, 3]"; primals_83: "f32[116]"; primals_84: "f32[116]"; primals_85: "f32[116, 116, 1, 1]"; primals_86: "f32[116]"; primals_87: "f32[116]"; primals_88: "f32[116, 116, 1, 1]"; primals_89: "f32[116]"; primals_90: "f32[116]"; primals_91: "f32[116, 1, 3, 3]"; primals_92: "f32[116]"; primals_93: "f32[116]"; primals_94: "f32[116, 116, 1, 1]"; primals_95: "f32[116]"; primals_96: "f32[116]"; primals_97: "f32[116, 116, 1, 1]"; primals_98: "f32[116]"; primals_99: "f32[116]"; primals_100: "f32[116, 1, 3, 3]"; primals_101: "f32[116]"; primals_102: "f32[116]"; primals_103: "f32[116, 116, 1, 1]"; primals_104: "f32[116]"; primals_105: "f32[116]"; primals_106: "f32[116, 116, 1, 1]"; primals_107: "f32[116]"; primals_108: "f32[116]"; primals_109: "f32[116, 1, 3, 3]"; primals_110: "f32[116]"; primals_111: "f32[116]"; primals_112: "f32[116, 116, 1, 1]"; primals_113: "f32[116]"; primals_114: "f32[116]"; primals_115: "f32[116, 116, 1, 1]"; primals_116: "f32[116]"; primals_117: "f32[116]"; primals_118: "f32[116, 1, 3, 3]"; primals_119: "f32[116]"; primals_120: "f32[116]"; primals_121: "f32[116, 116, 1, 1]"; primals_122: "f32[116]"; primals_123: "f32[116]"; primals_124: "f32[232, 1, 3, 3]"; primals_125: "f32[232]"; primals_126: "f32[232]"; primals_127: "f32[232, 232, 1, 1]"; primals_128: "f32[232]"; primals_129: "f32[232]"; primals_130: "f32[232, 232, 1, 1]"; primals_131: "f32[232]"; primals_132: "f32[232]"; primals_133: "f32[232, 1, 3, 3]"; primals_134: "f32[232]"; primals_135: "f32[232]"; primals_136: "f32[232, 232, 1, 1]"; primals_137: "f32[232]"; primals_138: "f32[232]"; primals_139: "f32[232, 232, 1, 1]"; primals_140: "f32[232]"; primals_141: "f32[232]"; primals_142: "f32[232, 1, 3, 3]"; primals_143: "f32[232]"; primals_144: "f32[232]"; primals_145: "f32[232, 232, 1, 1]"; primals_146: "f32[232]"; primals_147: "f32[232]"; primals_148: "f32[232, 232, 1, 1]"; primals_149: "f32[232]"; primals_150: "f32[232]"; primals_151: "f32[232, 1, 3, 3]"; primals_152: "f32[232]"; primals_153: "f32[232]"; primals_154: "f32[232, 232, 1, 1]"; primals_155: "f32[232]"; primals_156: "f32[232]"; primals_157: "f32[232, 232, 1, 1]"; primals_158: "f32[232]"; primals_159: "f32[232]"; primals_160: "f32[232, 1, 3, 3]"; primals_161: "f32[232]"; primals_162: "f32[232]"; primals_163: "f32[232, 232, 1, 1]"; primals_164: "f32[232]"; primals_165: "f32[232]"; primals_166: "f32[1024, 464, 1, 1]"; primals_167: "f32[1024]"; primals_168: "f32[1024]"; primals_169: "f32[1000, 1024]"; primals_170: "f32[1000]"; primals_171: "f32[24]"; primals_172: "f32[24]"; primals_173: "i64[]"; primals_174: "f32[24]"; primals_175: "f32[24]"; primals_176: "i64[]"; primals_177: "f32[58]"; primals_178: "f32[58]"; primals_179: "i64[]"; primals_180: "f32[58]"; primals_181: "f32[58]"; primals_182: "i64[]"; primals_183: "f32[58]"; primals_184: "f32[58]"; primals_185: "i64[]"; primals_186: "f32[58]"; primals_187: "f32[58]"; primals_188: "i64[]"; primals_189: "f32[58]"; primals_190: "f32[58]"; primals_191: "i64[]"; primals_192: "f32[58]"; primals_193: "f32[58]"; primals_194: "i64[]"; primals_195: "f32[58]"; primals_196: "f32[58]"; primals_197: "i64[]"; primals_198: "f32[58]"; primals_199: "f32[58]"; primals_200: "i64[]"; primals_201: "f32[58]"; primals_202: "f32[58]"; primals_203: "i64[]"; primals_204: "f32[58]"; primals_205: "f32[58]"; primals_206: "i64[]"; primals_207: "f32[58]"; primals_208: "f32[58]"; primals_209: "i64[]"; primals_210: "f32[58]"; primals_211: "f32[58]"; primals_212: "i64[]"; primals_213: "f32[58]"; primals_214: "f32[58]"; primals_215: "i64[]"; primals_216: "f32[116]"; primals_217: "f32[116]"; primals_218: "i64[]"; primals_219: "f32[116]"; primals_220: "f32[116]"; primals_221: "i64[]"; primals_222: "f32[116]"; primals_223: "f32[116]"; primals_224: "i64[]"; primals_225: "f32[116]"; primals_226: "f32[116]"; primals_227: "i64[]"; primals_228: "f32[116]"; primals_229: "f32[116]"; primals_230: "i64[]"; primals_231: "f32[116]"; primals_232: "f32[116]"; primals_233: "i64[]"; primals_234: "f32[116]"; primals_235: "f32[116]"; primals_236: "i64[]"; primals_237: "f32[116]"; primals_238: "f32[116]"; primals_239: "i64[]"; primals_240: "f32[116]"; primals_241: "f32[116]"; primals_242: "i64[]"; primals_243: "f32[116]"; primals_244: "f32[116]"; primals_245: "i64[]"; primals_246: "f32[116]"; primals_247: "f32[116]"; primals_248: "i64[]"; primals_249: "f32[116]"; primals_250: "f32[116]"; primals_251: "i64[]"; primals_252: "f32[116]"; primals_253: "f32[116]"; primals_254: "i64[]"; primals_255: "f32[116]"; primals_256: "f32[116]"; primals_257: "i64[]"; primals_258: "f32[116]"; primals_259: "f32[116]"; primals_260: "i64[]"; primals_261: "f32[116]"; primals_262: "f32[116]"; primals_263: "i64[]"; primals_264: "f32[116]"; primals_265: "f32[116]"; primals_266: "i64[]"; primals_267: "f32[116]"; primals_268: "f32[116]"; primals_269: "i64[]"; primals_270: "f32[116]"; primals_271: "f32[116]"; primals_272: "i64[]"; primals_273: "f32[116]"; primals_274: "f32[116]"; primals_275: "i64[]"; primals_276: "f32[116]"; primals_277: "f32[116]"; primals_278: "i64[]"; primals_279: "f32[116]"; primals_280: "f32[116]"; primals_281: "i64[]"; primals_282: "f32[116]"; primals_283: "f32[116]"; primals_284: "i64[]"; primals_285: "f32[116]"; primals_286: "f32[116]"; primals_287: "i64[]"; primals_288: "f32[116]"; primals_289: "f32[116]"; primals_290: "i64[]"; primals_291: "f32[116]"; primals_292: "f32[116]"; primals_293: "i64[]"; primals_294: "f32[232]"; primals_295: "f32[232]"; primals_296: "i64[]"; primals_297: "f32[232]"; primals_298: "f32[232]"; primals_299: "i64[]"; primals_300: "f32[232]"; primals_301: "f32[232]"; primals_302: "i64[]"; primals_303: "f32[232]"; primals_304: "f32[232]"; primals_305: "i64[]"; primals_306: "f32[232]"; primals_307: "f32[232]"; primals_308: "i64[]"; primals_309: "f32[232]"; primals_310: "f32[232]"; primals_311: "i64[]"; primals_312: "f32[232]"; primals_313: "f32[232]"; primals_314: "i64[]"; primals_315: "f32[232]"; primals_316: "f32[232]"; primals_317: "i64[]"; primals_318: "f32[232]"; primals_319: "f32[232]"; primals_320: "i64[]"; primals_321: "f32[232]"; primals_322: "f32[232]"; primals_323: "i64[]"; primals_324: "f32[232]"; primals_325: "f32[232]"; primals_326: "i64[]"; primals_327: "f32[232]"; primals_328: "f32[232]"; primals_329: "i64[]"; primals_330: "f32[232]"; primals_331: "f32[232]"; primals_332: "i64[]"; primals_333: "f32[232]"; primals_334: "f32[232]"; primals_335: "i64[]"; primals_336: "f32[1024]"; primals_337: "f32[1024]"; primals_338: "i64[]"; primals_339: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:155, code: x = self.conv1(x)
    convolution: "f32[4, 24, 112, 112]" = torch.ops.aten.convolution.default(primals_339, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_171, torch.float32)
    convert_element_type_1: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_172, torch.float32)
    add: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[24]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 24, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:156, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem: "f32[4, 24, 56, 56]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 24, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_1: "f32[4, 24, 28, 28]" = torch.ops.aten.convolution.default(getitem, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 24)
    convert_element_type_2: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_174, torch.float32)
    convert_element_type_3: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[24]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    convolution_2: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_3, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_4: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_177, torch.float32)
    convert_element_type_5: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    add_4: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[58]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    relu_1: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    convolution_3: "f32[4, 58, 56, 56]" = torch.ops.aten.convolution.default(getitem, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_6: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_180, torch.float32)
    convert_element_type_7: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    add_6: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[58]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 58, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 58, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    relu_2: "f32[4, 58, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    convolution_4: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 58)
    convert_element_type_8: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_183, torch.float32)
    convert_element_type_9: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    add_8: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[58]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    convolution_5: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_9, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_10: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_186, torch.float32)
    convert_element_type_11: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    add_10: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[58]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    relu_3: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    cat: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([relu_1, relu_3], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat, [4, 2, 58, 28, 28]);  cat = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3, 4]);  view = None
    clone: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_1: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone, [4, 116, 28, 28]);  clone = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split = torch.ops.aten.split.Tensor(view_1, 58, 1);  view_1 = None
    getitem_2: "f32[4, 58, 28, 28]" = split[0]
    getitem_3: "f32[4, 58, 28, 28]" = split[1];  split = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_6: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_3, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_12: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_189, torch.float32)
    convert_element_type_13: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    add_12: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[58]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_4: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    convolution_7: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
    convert_element_type_14: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_192, torch.float32)
    convert_element_type_15: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    add_14: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[58]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    convolution_8: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_15, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_16: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_195, torch.float32)
    convert_element_type_17: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    add_16: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[58]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    relu_5: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    cat_1: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_2, relu_5], 1);  getitem_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_2: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_1, [4, 2, 58, 28, 28]);  cat_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_1: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3, 4]);  view_2 = None
    clone_1: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_3: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_1, [4, 116, 28, 28]);  clone_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_1 = torch.ops.aten.split.Tensor(view_3, 58, 1);  view_3 = None
    getitem_4: "f32[4, 58, 28, 28]" = split_1[0]
    getitem_5: "f32[4, 58, 28, 28]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_9: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_5, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_18: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_198, torch.float32)
    convert_element_type_19: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_199, torch.float32)
    add_18: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[58]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    relu_6: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    convolution_10: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_6, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
    convert_element_type_20: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_201, torch.float32)
    convert_element_type_21: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_202, torch.float32)
    add_20: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[58]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_10: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_21: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    convolution_11: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_21, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_22: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_204, torch.float32)
    convert_element_type_23: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_205, torch.float32)
    add_22: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[58]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_11: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_23: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    relu_7: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    cat_2: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_4, relu_7], 1);  getitem_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_4: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_2, [4, 2, 58, 28, 28]);  cat_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_2: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3, 4]);  view_4 = None
    clone_2: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_5: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_2, [4, 116, 28, 28]);  clone_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_2 = torch.ops.aten.split.Tensor(view_5, 58, 1);  view_5 = None
    getitem_6: "f32[4, 58, 28, 28]" = split_2[0]
    getitem_7: "f32[4, 58, 28, 28]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_12: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(getitem_7, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_207, torch.float32)
    convert_element_type_25: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_208, torch.float32)
    add_24: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[58]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_12: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_25: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    relu_8: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    convolution_13: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
    convert_element_type_26: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_210, torch.float32)
    convert_element_type_27: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_211, torch.float32)
    add_26: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[58]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_13: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_27: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    convolution_14: "f32[4, 58, 28, 28]" = torch.ops.aten.convolution.default(add_27, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_213, torch.float32)
    convert_element_type_29: "f32[58]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    add_28: "f32[58]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[58]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_14: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_29: "f32[4, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    relu_9: "f32[4, 58, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    cat_3: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([getitem_6, relu_9], 1);  getitem_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_6: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat_3, [4, 2, 58, 28, 28]);  cat_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_3: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
    clone_3: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_7: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_3, [4, 116, 28, 28]);  clone_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_15: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(view_7, primals_46, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_30: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_216, torch.float32)
    convert_element_type_31: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    add_30: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[116]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_15: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_31: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    convolution_16: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_31, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_32: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_219, torch.float32)
    convert_element_type_33: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    add_32: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[116]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_16: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_33: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_10: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    convolution_17: "f32[4, 116, 28, 28]" = torch.ops.aten.convolution.default(view_7, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_222, torch.float32)
    convert_element_type_35: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    add_34: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[116]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_17: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 116, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_35: "f32[4, 116, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    relu_11: "f32[4, 116, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    convolution_18: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_11, primals_55, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_36: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_225, torch.float32)
    convert_element_type_37: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    add_36: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[116]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_18: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_37: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    convolution_19: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_37, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_38: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_228, torch.float32)
    convert_element_type_39: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    add_38: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[116]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_19: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_39: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    relu_12: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    cat_4: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([relu_10, relu_12], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_8: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_4, [4, 2, 116, 14, 14]);  cat_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_4: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3, 4]);  view_8 = None
    clone_4: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_9: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_4, [4, 232, 14, 14]);  clone_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_3 = torch.ops.aten.split.Tensor(view_9, 116, 1);  view_9 = None
    getitem_8: "f32[4, 116, 14, 14]" = split_3[0]
    getitem_9: "f32[4, 116, 14, 14]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_20: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_9, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_40: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_231, torch.float32)
    convert_element_type_41: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    add_40: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[116]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_20: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_165: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_167: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_41: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    relu_13: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    convolution_21: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_13, primals_64, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_42: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_234, torch.float32)
    convert_element_type_43: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    add_42: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[116]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_21: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_173: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_175: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_43: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    convolution_22: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_43, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_44: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_237, torch.float32)
    convert_element_type_45: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    add_44: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[116]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_22: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_181: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_183: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_45: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    relu_14: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    cat_5: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_8, relu_14], 1);  getitem_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_10: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_5, [4, 2, 116, 14, 14]);  cat_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_5: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3, 4]);  view_10 = None
    clone_5: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_11: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_5, [4, 232, 14, 14]);  clone_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_4 = torch.ops.aten.split.Tensor(view_11, 116, 1);  view_11 = None
    getitem_10: "f32[4, 116, 14, 14]" = split_4[0]
    getitem_11: "f32[4, 116, 14, 14]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_23: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_11, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_46: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_240, torch.float32)
    convert_element_type_47: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    add_46: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[116]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_23: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_189: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_191: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_47: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    relu_15: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    convolution_24: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_48: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_243, torch.float32)
    convert_element_type_49: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    add_48: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[116]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_24: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_197: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_199: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_49: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    convolution_25: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_49, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_50: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_246, torch.float32)
    convert_element_type_51: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    add_50: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[116]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_25: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_205: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_207: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_51: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    relu_16: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    cat_6: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_10, relu_16], 1);  getitem_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_12: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_6, [4, 2, 116, 14, 14]);  cat_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_6: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3, 4]);  view_12 = None
    clone_6: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_13: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_6, [4, 232, 14, 14]);  clone_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_5 = torch.ops.aten.split.Tensor(view_13, 116, 1);  view_13 = None
    getitem_12: "f32[4, 116, 14, 14]" = split_5[0]
    getitem_13: "f32[4, 116, 14, 14]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_26: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_13, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_52: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_249, torch.float32)
    convert_element_type_53: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    add_52: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[116]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_26: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_213: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_215: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_53: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    relu_17: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    convolution_27: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_54: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_252, torch.float32)
    convert_element_type_55: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    add_54: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[116]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_27: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_221: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_223: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_55: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    convolution_28: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_55, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_56: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_255, torch.float32)
    convert_element_type_57: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    add_56: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[116]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_28: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_229: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_231: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_57: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    relu_18: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    cat_7: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_12, relu_18], 1);  getitem_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_14: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_7, [4, 2, 116, 14, 14]);  cat_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_7: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3, 4]);  view_14 = None
    clone_7: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_15: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_7, [4, 232, 14, 14]);  clone_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_6 = torch.ops.aten.split.Tensor(view_15, 116, 1);  view_15 = None
    getitem_14: "f32[4, 116, 14, 14]" = split_6[0]
    getitem_15: "f32[4, 116, 14, 14]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_29: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_15, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_58: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_258, torch.float32)
    convert_element_type_59: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    add_58: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[116]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_29: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_237: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_239: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_59: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    relu_19: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    convolution_30: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_19, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_60: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_261, torch.float32)
    convert_element_type_61: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    add_60: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[116]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_30: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_245: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_247: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_61: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    convolution_31: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_61, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_62: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_264, torch.float32)
    convert_element_type_63: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    add_62: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[116]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_31: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_253: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_255: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_63: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    relu_20: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    cat_8: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_14, relu_20], 1);  getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_16: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_8, [4, 2, 116, 14, 14]);  cat_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_8: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3, 4]);  view_16 = None
    clone_8: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_17: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_8, [4, 232, 14, 14]);  clone_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_7 = torch.ops.aten.split.Tensor(view_17, 116, 1);  view_17 = None
    getitem_16: "f32[4, 116, 14, 14]" = split_7[0]
    getitem_17: "f32[4, 116, 14, 14]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_32: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_17, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_64: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    convert_element_type_65: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    add_64: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[116]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_32: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_261: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_263: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_65: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    relu_21: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    convolution_33: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_66: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    convert_element_type_67: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    add_66: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[116]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_33: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_269: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_271: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_67: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    convolution_34: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_67, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_68: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    convert_element_type_69: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    add_68: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[116]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_34: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_277: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_279: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_69: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    relu_22: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    cat_9: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_16, relu_22], 1);  getitem_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_18: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_9, [4, 2, 116, 14, 14]);  cat_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_9: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3, 4]);  view_18 = None
    clone_9: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_19: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_9, [4, 232, 14, 14]);  clone_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_8 = torch.ops.aten.split.Tensor(view_19, 116, 1);  view_19 = None
    getitem_18: "f32[4, 116, 14, 14]" = split_8[0]
    getitem_19: "f32[4, 116, 14, 14]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_35: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_19, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_70: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    convert_element_type_71: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    add_70: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[116]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_35: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_285: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_287: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_71: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    relu_23: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    convolution_36: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_72: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    convert_element_type_73: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    add_72: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[116]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_36: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_293: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_295: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_73: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    convolution_37: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_73, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_74: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    convert_element_type_75: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    add_74: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[116]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_37: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_301: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_303: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_75: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    relu_24: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_75);  add_75 = None
    cat_10: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_18, relu_24], 1);  getitem_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_20: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_10, [4, 2, 116, 14, 14]);  cat_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_10: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3, 4]);  view_20 = None
    clone_10: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_21: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_10, [4, 232, 14, 14]);  clone_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_9 = torch.ops.aten.split.Tensor(view_21, 116, 1);  view_21 = None
    getitem_20: "f32[4, 116, 14, 14]" = split_9[0]
    getitem_21: "f32[4, 116, 14, 14]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_38: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(getitem_21, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_76: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    convert_element_type_77: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    add_76: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[116]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_38: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_309: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_311: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_77: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    relu_25: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    convolution_39: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
    convert_element_type_78: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    convert_element_type_79: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    add_78: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[116]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_39: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  unsqueeze_313 = None
    mul_118: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_317: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_319: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_79: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    convolution_40: "f32[4, 116, 14, 14]" = torch.ops.aten.convolution.default(add_79, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_80: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    convert_element_type_81: "f32[116]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    add_80: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[116]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_40: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  unsqueeze_321 = None
    mul_121: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_325: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_327: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_81: "f32[4, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    relu_26: "f32[4, 116, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    cat_11: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_20, relu_26], 1);  getitem_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_22: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat_11, [4, 2, 116, 14, 14]);  cat_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_11: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3, 4]);  view_22 = None
    clone_11: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_23: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_11, [4, 232, 14, 14]);  clone_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    convolution_41: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(view_23, primals_124, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232)
    convert_element_type_82: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    convert_element_type_83: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    add_82: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[232]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_41: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  unsqueeze_329 = None
    mul_124: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_333: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_335: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_83: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    convolution_42: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_83, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_84: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    convert_element_type_85: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    add_84: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[232]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_42: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  unsqueeze_337 = None
    mul_127: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_341: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_343: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_85: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    relu_27: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    convolution_43: "f32[4, 232, 14, 14]" = torch.ops.aten.convolution.default(view_23, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_86: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    convert_element_type_87: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    add_86: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[232]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_43: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 232, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  unsqueeze_345 = None
    mul_130: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_349: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_351: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_87: "f32[4, 232, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    relu_28: "f32[4, 232, 14, 14]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    convolution_44: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232)
    convert_element_type_88: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    convert_element_type_89: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    add_88: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[232]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_44: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  unsqueeze_353 = None
    mul_133: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_357: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_359: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_89: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    convolution_45: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_89, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_90: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    convert_element_type_91: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    add_90: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[232]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_45: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  unsqueeze_361 = None
    mul_136: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_365: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_367: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_91: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    relu_29: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    cat_12: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([relu_27, relu_29], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_24: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_12, [4, 2, 232, 7, 7]);  cat_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_12: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
    clone_12: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_25: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_12, [4, 464, 7, 7]);  clone_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_10 = torch.ops.aten.split.Tensor(view_25, 232, 1);  view_25 = None
    getitem_22: "f32[4, 232, 7, 7]" = split_10[0]
    getitem_23: "f32[4, 232, 7, 7]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_46: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_23, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_92: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    convert_element_type_93: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    add_92: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[232]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_46: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  unsqueeze_369 = None
    mul_139: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_373: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_375: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_93: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    relu_30: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    convolution_47: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
    convert_element_type_94: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_312, torch.float32)
    convert_element_type_95: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_313, torch.float32)
    add_94: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[232]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_47: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  unsqueeze_377 = None
    mul_142: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_381: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_383: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_95: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    convolution_48: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_95, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_96: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_315, torch.float32)
    convert_element_type_97: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_316, torch.float32)
    add_96: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[232]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_48: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  unsqueeze_385 = None
    mul_145: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_389: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_391: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_97: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    relu_31: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    cat_13: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_22, relu_31], 1);  getitem_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_26: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_13, [4, 2, 232, 7, 7]);  cat_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_13: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3, 4]);  view_26 = None
    clone_13: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_27: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_13, [4, 464, 7, 7]);  clone_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_11 = torch.ops.aten.split.Tensor(view_27, 232, 1);  view_27 = None
    getitem_24: "f32[4, 232, 7, 7]" = split_11[0]
    getitem_25: "f32[4, 232, 7, 7]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_49: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_25, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_98: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_318, torch.float32)
    convert_element_type_99: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_319, torch.float32)
    add_98: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[232]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_49: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  unsqueeze_393 = None
    mul_148: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_397: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_399: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_99: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    relu_32: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    convolution_50: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
    convert_element_type_100: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_321, torch.float32)
    convert_element_type_101: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_322, torch.float32)
    add_100: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[232]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_50: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  unsqueeze_401 = None
    mul_151: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_405: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_407: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_101: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    convolution_51: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_101, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_102: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_324, torch.float32)
    convert_element_type_103: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_325, torch.float32)
    add_102: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[232]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_51: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  unsqueeze_409 = None
    mul_154: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_413: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_415: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_103: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    relu_33: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_103);  add_103 = None
    cat_14: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_24, relu_33], 1);  getitem_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_28: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_14, [4, 2, 232, 7, 7]);  cat_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_14: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3, 4]);  view_28 = None
    clone_14: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_29: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_14, [4, 464, 7, 7]);  clone_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    split_12 = torch.ops.aten.split.Tensor(view_29, 232, 1);  view_29 = None
    getitem_26: "f32[4, 232, 7, 7]" = split_12[0]
    getitem_27: "f32[4, 232, 7, 7]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    convolution_52: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(getitem_27, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_104: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_327, torch.float32)
    convert_element_type_105: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_328, torch.float32)
    add_104: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[232]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
    reciprocal_52: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  unsqueeze_417 = None
    mul_157: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_421: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_423: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_105: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    relu_34: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    convolution_53: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(relu_34, primals_160, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
    convert_element_type_106: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_330, torch.float32)
    convert_element_type_107: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_331, torch.float32)
    add_106: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[232]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_53: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  unsqueeze_425 = None
    mul_160: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_429: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_431: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_107: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    convolution_54: "f32[4, 232, 7, 7]" = torch.ops.aten.convolution.default(add_107, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_108: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_333, torch.float32)
    convert_element_type_109: "f32[232]" = torch.ops.prims.convert_element_type.default(primals_334, torch.float32)
    add_108: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[232]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_54: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  unsqueeze_433 = None
    mul_163: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_437: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_439: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_109: "f32[4, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    relu_35: "f32[4, 232, 7, 7]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    cat_15: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_26, relu_35], 1);  getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    view_30: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_15, [4, 2, 232, 7, 7]);  cat_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_15: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3, 4]);  view_30 = None
    clone_15: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_31: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_15, [4, 464, 7, 7]);  clone_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:160, code: x = self.conv5(x)
    convolution_55: "f32[4, 1024, 7, 7]" = torch.ops.aten.convolution.default(view_31, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_110: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_336, torch.float32)
    convert_element_type_111: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_337, torch.float32)
    add_110: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[1024]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_55: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  unsqueeze_441 = None
    mul_166: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_445: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_447: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_111: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    relu_36: "f32[4, 1024, 7, 7]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:161, code: x = x.mean([2, 3])  # globalpool
    mean: "f32[4, 1024]" = torch.ops.aten.mean.dim(relu_36, [2, 3])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:162, code: x = self.fc(x)
    permute_16: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_170, mean, permute_16);  primals_170 = None
    permute_17: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm: "f32[4, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_17);  permute_17 = None
    permute_18: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_18, mean);  permute_18 = mean = None
    permute_19: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_32: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_20: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:161, code: x = x.mean([2, 3])  # globalpool
    unsqueeze_448: "f32[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
    unsqueeze_449: "f32[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    expand: "f32[4, 1024, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_449, [4, 1024, 7, 7]);  unsqueeze_449 = None
    div: "f32[4, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:160, code: x = self.conv5(x)
    alias_38: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_39: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    add_112: "f32[1024]" = torch.ops.aten.add.Tensor(primals_337, 1e-05);  primals_337 = None
    rsqrt: "f32[1024]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    unsqueeze_450: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_336, 0);  primals_336 = None
    unsqueeze_451: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_56: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_452);  convolution_55 = unsqueeze_452 = None
    mul_168: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_56);  sub_56 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 2, 3]);  mul_168 = None
    mul_173: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt, primals_167);  primals_167 = None
    unsqueeze_459: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_173, 0);  mul_173 = None
    unsqueeze_460: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_174: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_461);  where = unsqueeze_461 = None
    mul_175: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_174, view_31, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_174 = view_31 = primals_166 = None
    getitem_28: "f32[4, 464, 7, 7]" = convolution_backward[0]
    getitem_29: "f32[1024, 464, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_33: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.view.default(getitem_28, [4, 232, 2, 7, 7]);  getitem_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_21: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_16: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_34: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_16, [4, 464, 7, 7]);  clone_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_1: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_34, 1, 0, 232)
    slice_2: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_34, 1, 232, 464);  view_34 = None
    alias_41: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_42: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_1: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_42, 0);  alias_42 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, slice_2);  le_1 = scalar_tensor_1 = slice_2 = None
    add_113: "f32[232]" = torch.ops.aten.add.Tensor(primals_334, 1e-05);  primals_334 = None
    rsqrt_1: "f32[232]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    unsqueeze_462: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_333, 0);  primals_333 = None
    unsqueeze_463: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    sum_4: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_57: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_464);  convolution_54 = unsqueeze_464 = None
    mul_176: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_57);  sub_57 = None
    sum_5: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 2, 3]);  mul_176 = None
    mul_181: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_164);  primals_164 = None
    unsqueeze_471: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
    unsqueeze_472: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_182: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_473);  where_1 = unsqueeze_473 = None
    mul_183: "f32[232]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_182, add_107, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_182 = add_107 = primals_163 = None
    getitem_31: "f32[4, 232, 7, 7]" = convolution_backward_1[0]
    getitem_32: "f32[232, 232, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    add_114: "f32[232]" = torch.ops.aten.add.Tensor(primals_331, 1e-05);  primals_331 = None
    rsqrt_2: "f32[232]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    unsqueeze_474: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_330, 0);  primals_330 = None
    unsqueeze_475: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    sum_6: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_31, [0, 2, 3])
    sub_58: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_476);  convolution_53 = unsqueeze_476 = None
    mul_184: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_31, sub_58);  sub_58 = None
    sum_7: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 2, 3]);  mul_184 = None
    mul_189: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_161);  primals_161 = None
    unsqueeze_483: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_189, 0);  mul_189 = None
    unsqueeze_484: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_190: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_31, unsqueeze_485);  getitem_31 = unsqueeze_485 = None
    mul_191: "f32[232]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_190, relu_34, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_190 = primals_160 = None
    getitem_34: "f32[4, 232, 7, 7]" = convolution_backward_2[0]
    getitem_35: "f32[232, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    alias_44: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_45: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_2: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_34);  le_2 = scalar_tensor_2 = getitem_34 = None
    add_115: "f32[232]" = torch.ops.aten.add.Tensor(primals_328, 1e-05);  primals_328 = None
    rsqrt_3: "f32[232]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    unsqueeze_486: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_327, 0);  primals_327 = None
    unsqueeze_487: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    sum_8: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_59: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_488);  convolution_52 = unsqueeze_488 = None
    mul_192: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_59);  sub_59 = None
    sum_9: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 2, 3]);  mul_192 = None
    mul_197: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_158);  primals_158 = None
    unsqueeze_495: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_197, 0);  mul_197 = None
    unsqueeze_496: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_198: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_497);  where_2 = unsqueeze_497 = None
    mul_199: "f32[232]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_198, getitem_27, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_198 = getitem_27 = primals_157 = None
    getitem_37: "f32[4, 232, 7, 7]" = convolution_backward_3[0]
    getitem_38: "f32[232, 232, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_16: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([slice_1, getitem_37], 1);  slice_1 = getitem_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_35: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.view.default(cat_16, [4, 232, 2, 7, 7]);  cat_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_22: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3, 4]);  view_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_17: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
    view_36: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_17, [4, 464, 7, 7]);  clone_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_3: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_36, 1, 0, 232)
    slice_4: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_36, 1, 232, 464);  view_36 = None
    alias_47: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_48: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_3: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, slice_4);  le_3 = scalar_tensor_3 = slice_4 = None
    add_116: "f32[232]" = torch.ops.aten.add.Tensor(primals_325, 1e-05);  primals_325 = None
    rsqrt_4: "f32[232]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_498: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_324, 0);  primals_324 = None
    unsqueeze_499: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    sum_10: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_60: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_500);  convolution_51 = unsqueeze_500 = None
    mul_200: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_60);  sub_60 = None
    sum_11: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 2, 3]);  mul_200 = None
    mul_205: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_155);  primals_155 = None
    unsqueeze_507: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_205, 0);  mul_205 = None
    unsqueeze_508: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_206: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_509);  where_3 = unsqueeze_509 = None
    mul_207: "f32[232]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_206, add_101, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_206 = add_101 = primals_154 = None
    getitem_40: "f32[4, 232, 7, 7]" = convolution_backward_4[0]
    getitem_41: "f32[232, 232, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    add_117: "f32[232]" = torch.ops.aten.add.Tensor(primals_322, 1e-05);  primals_322 = None
    rsqrt_5: "f32[232]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    unsqueeze_510: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_321, 0);  primals_321 = None
    unsqueeze_511: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    sum_12: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_40, [0, 2, 3])
    sub_61: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_512);  convolution_50 = unsqueeze_512 = None
    mul_208: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_40, sub_61);  sub_61 = None
    sum_13: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 2, 3]);  mul_208 = None
    mul_213: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_152);  primals_152 = None
    unsqueeze_519: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_213, 0);  mul_213 = None
    unsqueeze_520: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_214: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_40, unsqueeze_521);  getitem_40 = unsqueeze_521 = None
    mul_215: "f32[232]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_214, relu_32, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_214 = primals_151 = None
    getitem_43: "f32[4, 232, 7, 7]" = convolution_backward_5[0]
    getitem_44: "f32[232, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    alias_50: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_51: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_4: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_43);  le_4 = scalar_tensor_4 = getitem_43 = None
    add_118: "f32[232]" = torch.ops.aten.add.Tensor(primals_319, 1e-05);  primals_319 = None
    rsqrt_6: "f32[232]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    unsqueeze_522: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_318, 0);  primals_318 = None
    unsqueeze_523: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    sum_14: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_62: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_524);  convolution_49 = unsqueeze_524 = None
    mul_216: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_62);  sub_62 = None
    sum_15: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3]);  mul_216 = None
    mul_221: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_149);  primals_149 = None
    unsqueeze_531: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
    unsqueeze_532: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_222: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_533);  where_4 = unsqueeze_533 = None
    mul_223: "f32[232]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_222, getitem_25, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_222 = getitem_25 = primals_148 = None
    getitem_46: "f32[4, 232, 7, 7]" = convolution_backward_6[0]
    getitem_47: "f32[232, 232, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_17: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([slice_3, getitem_46], 1);  slice_3 = getitem_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_37: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.view.default(cat_17, [4, 232, 2, 7, 7]);  cat_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_23: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3, 4]);  view_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_18: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_38: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_18, [4, 464, 7, 7]);  clone_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_5: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_38, 1, 0, 232)
    slice_6: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_38, 1, 232, 464);  view_38 = None
    alias_53: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_54: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_5: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_54, 0);  alias_54 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, slice_6);  le_5 = scalar_tensor_5 = slice_6 = None
    add_119: "f32[232]" = torch.ops.aten.add.Tensor(primals_316, 1e-05);  primals_316 = None
    rsqrt_7: "f32[232]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    unsqueeze_534: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_315, 0);  primals_315 = None
    unsqueeze_535: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    sum_16: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_63: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_536);  convolution_48 = unsqueeze_536 = None
    mul_224: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_63);  sub_63 = None
    sum_17: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 2, 3]);  mul_224 = None
    mul_229: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_146);  primals_146 = None
    unsqueeze_543: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_229, 0);  mul_229 = None
    unsqueeze_544: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_230: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_545);  where_5 = unsqueeze_545 = None
    mul_231: "f32[232]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_230, add_95, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_230 = add_95 = primals_145 = None
    getitem_49: "f32[4, 232, 7, 7]" = convolution_backward_7[0]
    getitem_50: "f32[232, 232, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    add_120: "f32[232]" = torch.ops.aten.add.Tensor(primals_313, 1e-05);  primals_313 = None
    rsqrt_8: "f32[232]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_546: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_312, 0);  primals_312 = None
    unsqueeze_547: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    sum_18: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_49, [0, 2, 3])
    sub_64: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_548);  convolution_47 = unsqueeze_548 = None
    mul_232: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_49, sub_64);  sub_64 = None
    sum_19: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 2, 3]);  mul_232 = None
    mul_237: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_143);  primals_143 = None
    unsqueeze_555: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_237, 0);  mul_237 = None
    unsqueeze_556: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_238: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_49, unsqueeze_557);  getitem_49 = unsqueeze_557 = None
    mul_239: "f32[232]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_238, relu_30, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_238 = primals_142 = None
    getitem_52: "f32[4, 232, 7, 7]" = convolution_backward_8[0]
    getitem_53: "f32[232, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    alias_56: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_57: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_6: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_57, 0);  alias_57 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_52);  le_6 = scalar_tensor_6 = getitem_52 = None
    add_121: "f32[232]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_9: "f32[232]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    unsqueeze_558: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_559: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    sum_20: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_65: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_560);  convolution_46 = unsqueeze_560 = None
    mul_240: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_65);  sub_65 = None
    sum_21: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 2, 3]);  mul_240 = None
    mul_245: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_140);  primals_140 = None
    unsqueeze_567: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
    unsqueeze_568: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_246: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_569);  where_6 = unsqueeze_569 = None
    mul_247: "f32[232]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_246, getitem_23, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_246 = getitem_23 = primals_139 = None
    getitem_55: "f32[4, 232, 7, 7]" = convolution_backward_9[0]
    getitem_56: "f32[232, 232, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_18: "f32[4, 464, 7, 7]" = torch.ops.aten.cat.default([slice_5, getitem_55], 1);  slice_5 = getitem_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_39: "f32[4, 232, 2, 7, 7]" = torch.ops.aten.view.default(cat_18, [4, 232, 2, 7, 7]);  cat_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_24: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_19: "f32[4, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    view_40: "f32[4, 464, 7, 7]" = torch.ops.aten.view.default(clone_19, [4, 464, 7, 7]);  clone_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    slice_7: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_40, 1, 0, 232)
    slice_8: "f32[4, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_40, 1, 232, 464);  view_40 = None
    alias_59: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_60: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_7: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_60, 0);  alias_60 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, slice_8);  le_7 = scalar_tensor_7 = slice_8 = None
    add_122: "f32[232]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_10: "f32[232]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_570: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_571: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    sum_22: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_66: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_572);  convolution_45 = unsqueeze_572 = None
    mul_248: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_66);  sub_66 = None
    sum_23: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 2, 3]);  mul_248 = None
    mul_253: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_137);  primals_137 = None
    unsqueeze_579: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
    unsqueeze_580: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_254: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_581);  where_7 = unsqueeze_581 = None
    mul_255: "f32[232]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_254, add_89, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_254 = add_89 = primals_136 = None
    getitem_58: "f32[4, 232, 7, 7]" = convolution_backward_10[0]
    getitem_59: "f32[232, 232, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    add_123: "f32[232]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_11: "f32[232]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_582: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_583: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    sum_24: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_58, [0, 2, 3])
    sub_67: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_584);  convolution_44 = unsqueeze_584 = None
    mul_256: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_58, sub_67);  sub_67 = None
    sum_25: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 2, 3]);  mul_256 = None
    mul_261: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_134);  primals_134 = None
    unsqueeze_591: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
    unsqueeze_592: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_262: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_58, unsqueeze_593);  getitem_58 = unsqueeze_593 = None
    mul_263: "f32[232]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_262, relu_28, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_262 = primals_133 = None
    getitem_61: "f32[4, 232, 14, 14]" = convolution_backward_11[0]
    getitem_62: "f32[232, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    alias_62: "f32[4, 232, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_63: "f32[4, 232, 14, 14]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_8: "b8[4, 232, 14, 14]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[4, 232, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_61);  le_8 = scalar_tensor_8 = getitem_61 = None
    add_124: "f32[232]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_12: "f32[232]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_594: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_595: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    sum_26: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_68: "f32[4, 232, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_596);  convolution_43 = unsqueeze_596 = None
    mul_264: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_68);  sub_68 = None
    sum_27: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
    mul_269: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_131);  primals_131 = None
    unsqueeze_603: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_604: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_270: "f32[4, 232, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_605);  where_8 = unsqueeze_605 = None
    mul_271: "f32[232]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_270, view_23, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_270 = primals_130 = None
    getitem_64: "f32[4, 232, 14, 14]" = convolution_backward_12[0]
    getitem_65: "f32[232, 232, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    alias_65: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_66: "f32[4, 232, 7, 7]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_9: "b8[4, 232, 7, 7]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 232, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, slice_7);  le_9 = scalar_tensor_9 = slice_7 = None
    add_125: "f32[232]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_13: "f32[232]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    unsqueeze_606: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_607: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    sum_28: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_69: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_608);  convolution_42 = unsqueeze_608 = None
    mul_272: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_69);  sub_69 = None
    sum_29: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 2, 3]);  mul_272 = None
    mul_277: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_128);  primals_128 = None
    unsqueeze_615: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
    unsqueeze_616: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_278: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_617);  where_9 = unsqueeze_617 = None
    mul_279: "f32[232]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_278, add_83, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_278 = add_83 = primals_127 = None
    getitem_67: "f32[4, 232, 7, 7]" = convolution_backward_13[0]
    getitem_68: "f32[232, 232, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    add_126: "f32[232]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_14: "f32[232]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    unsqueeze_618: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_619: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    sum_30: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_67, [0, 2, 3])
    sub_70: "f32[4, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_620);  convolution_41 = unsqueeze_620 = None
    mul_280: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_67, sub_70);  sub_70 = None
    sum_31: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 2, 3]);  mul_280 = None
    mul_285: "f32[232]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_125);  primals_125 = None
    unsqueeze_627: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
    unsqueeze_628: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_286: "f32[4, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_67, unsqueeze_629);  getitem_67 = unsqueeze_629 = None
    mul_287: "f32[232]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_286, view_23, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_286 = view_23 = primals_124 = None
    getitem_70: "f32[4, 232, 14, 14]" = convolution_backward_14[0]
    getitem_71: "f32[232, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    add_127: "f32[4, 232, 14, 14]" = torch.ops.aten.add.Tensor(getitem_64, getitem_70);  getitem_64 = getitem_70 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_41: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(add_127, [4, 116, 2, 14, 14]);  add_127 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_25: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_20: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_42: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_20, [4, 232, 14, 14]);  clone_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_9: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_42, 1, 0, 116)
    slice_10: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_42, 1, 116, 232);  view_42 = None
    alias_68: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_69: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_10: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, slice_10);  le_10 = scalar_tensor_10 = slice_10 = None
    add_128: "f32[116]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_15: "f32[116]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_630: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_631: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    sum_32: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_71: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_632);  convolution_40 = unsqueeze_632 = None
    mul_288: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_71);  sub_71 = None
    sum_33: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 2, 3]);  mul_288 = None
    mul_293: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_122);  primals_122 = None
    unsqueeze_639: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_293, 0);  mul_293 = None
    unsqueeze_640: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_294: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_641);  where_10 = unsqueeze_641 = None
    mul_295: "f32[116]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_294, add_79, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_294 = add_79 = primals_121 = None
    getitem_73: "f32[4, 116, 14, 14]" = convolution_backward_15[0]
    getitem_74: "f32[116, 116, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    add_129: "f32[116]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_16: "f32[116]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    unsqueeze_642: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_643: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    sum_34: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_73, [0, 2, 3])
    sub_72: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_644);  convolution_39 = unsqueeze_644 = None
    mul_296: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_73, sub_72);  sub_72 = None
    sum_35: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 2, 3]);  mul_296 = None
    mul_301: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_119);  primals_119 = None
    unsqueeze_651: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_652: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_302: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_73, unsqueeze_653);  getitem_73 = unsqueeze_653 = None
    mul_303: "f32[116]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_302, relu_25, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_302 = primals_118 = None
    getitem_76: "f32[4, 116, 14, 14]" = convolution_backward_16[0]
    getitem_77: "f32[116, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    alias_71: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_72: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_11: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_72, 0);  alias_72 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_76);  le_11 = scalar_tensor_11 = getitem_76 = None
    add_130: "f32[116]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_17: "f32[116]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_654: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_655: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    sum_36: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_73: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_656);  convolution_38 = unsqueeze_656 = None
    mul_304: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_73);  sub_73 = None
    sum_37: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 2, 3]);  mul_304 = None
    mul_309: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_116);  primals_116 = None
    unsqueeze_663: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_309, 0);  mul_309 = None
    unsqueeze_664: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_310: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_665);  where_11 = unsqueeze_665 = None
    mul_311: "f32[116]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_310, getitem_21, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = getitem_21 = primals_115 = None
    getitem_79: "f32[4, 116, 14, 14]" = convolution_backward_17[0]
    getitem_80: "f32[116, 116, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_19: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_9, getitem_79], 1);  slice_9 = getitem_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_43: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_19, [4, 116, 2, 14, 14]);  cat_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_26: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_21: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    view_44: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_21, [4, 232, 14, 14]);  clone_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_11: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_44, 1, 0, 116)
    slice_12: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_44, 1, 116, 232);  view_44 = None
    alias_74: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_75: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_12: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, slice_12);  le_12 = scalar_tensor_12 = slice_12 = None
    add_131: "f32[116]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_18: "f32[116]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_666: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_667: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    sum_38: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_74: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_668);  convolution_37 = unsqueeze_668 = None
    mul_312: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_74);  sub_74 = None
    sum_39: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 2, 3]);  mul_312 = None
    mul_317: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_113);  primals_113 = None
    unsqueeze_675: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
    unsqueeze_676: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_318: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_677);  where_12 = unsqueeze_677 = None
    mul_319: "f32[116]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_318, add_73, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_318 = add_73 = primals_112 = None
    getitem_82: "f32[4, 116, 14, 14]" = convolution_backward_18[0]
    getitem_83: "f32[116, 116, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    add_132: "f32[116]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_19: "f32[116]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_678: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_679: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    sum_40: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_82, [0, 2, 3])
    sub_75: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_680);  convolution_36 = unsqueeze_680 = None
    mul_320: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_82, sub_75);  sub_75 = None
    sum_41: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3]);  mul_320 = None
    mul_325: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_110);  primals_110 = None
    unsqueeze_687: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
    unsqueeze_688: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_326: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_82, unsqueeze_689);  getitem_82 = unsqueeze_689 = None
    mul_327: "f32[116]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_326, relu_23, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_326 = primals_109 = None
    getitem_85: "f32[4, 116, 14, 14]" = convolution_backward_19[0]
    getitem_86: "f32[116, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    alias_77: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_78: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    le_13: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_78, 0);  alias_78 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_85);  le_13 = scalar_tensor_13 = getitem_85 = None
    add_133: "f32[116]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_20: "f32[116]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_690: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_691: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    sum_42: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_76: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_692);  convolution_35 = unsqueeze_692 = None
    mul_328: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_76);  sub_76 = None
    sum_43: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 2, 3]);  mul_328 = None
    mul_333: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_107);  primals_107 = None
    unsqueeze_699: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_700: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_334: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_701);  where_13 = unsqueeze_701 = None
    mul_335: "f32[116]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_334, getitem_19, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = getitem_19 = primals_106 = None
    getitem_88: "f32[4, 116, 14, 14]" = convolution_backward_20[0]
    getitem_89: "f32[116, 116, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_20: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_11, getitem_88], 1);  slice_11 = getitem_88 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_45: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_20, [4, 116, 2, 14, 14]);  cat_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_27: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3, 4]);  view_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_22: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_46: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_22, [4, 232, 14, 14]);  clone_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_13: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_46, 1, 0, 116)
    slice_14: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_46, 1, 116, 232);  view_46 = None
    alias_80: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_81: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_14: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, slice_14);  le_14 = scalar_tensor_14 = slice_14 = None
    add_134: "f32[116]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_21: "f32[116]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_702: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_703: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    sum_44: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_77: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_704);  convolution_34 = unsqueeze_704 = None
    mul_336: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_77);  sub_77 = None
    sum_45: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3]);  mul_336 = None
    mul_341: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_104);  primals_104 = None
    unsqueeze_711: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_712: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_342: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_713);  where_14 = unsqueeze_713 = None
    mul_343: "f32[116]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_342, add_67, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_342 = add_67 = primals_103 = None
    getitem_91: "f32[4, 116, 14, 14]" = convolution_backward_21[0]
    getitem_92: "f32[116, 116, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    add_135: "f32[116]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_22: "f32[116]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    unsqueeze_714: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_715: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    sum_46: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_91, [0, 2, 3])
    sub_78: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_716);  convolution_33 = unsqueeze_716 = None
    mul_344: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_91, sub_78);  sub_78 = None
    sum_47: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 2, 3]);  mul_344 = None
    mul_349: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_101);  primals_101 = None
    unsqueeze_723: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
    unsqueeze_724: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_350: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_91, unsqueeze_725);  getitem_91 = unsqueeze_725 = None
    mul_351: "f32[116]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_350, relu_21, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_350 = primals_100 = None
    getitem_94: "f32[4, 116, 14, 14]" = convolution_backward_22[0]
    getitem_95: "f32[116, 1, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    alias_83: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_84: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_15: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_84, 0);  alias_84 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_94);  le_15 = scalar_tensor_15 = getitem_94 = None
    add_136: "f32[116]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_23: "f32[116]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_726: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_727: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    sum_48: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_79: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_728);  convolution_32 = unsqueeze_728 = None
    mul_352: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_79);  sub_79 = None
    sum_49: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_357: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_98);  primals_98 = None
    unsqueeze_735: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_736: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_358: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_737);  where_15 = unsqueeze_737 = None
    mul_359: "f32[116]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_358, getitem_17, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_358 = getitem_17 = primals_97 = None
    getitem_97: "f32[4, 116, 14, 14]" = convolution_backward_23[0]
    getitem_98: "f32[116, 116, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_21: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_13, getitem_97], 1);  slice_13 = getitem_97 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_47: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_21, [4, 116, 2, 14, 14]);  cat_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_28: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3, 4]);  view_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_23: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_48: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_23, [4, 232, 14, 14]);  clone_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_15: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_48, 1, 0, 116)
    slice_16: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_48, 1, 116, 232);  view_48 = None
    alias_86: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_87: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_16: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, slice_16);  le_16 = scalar_tensor_16 = slice_16 = None
    add_137: "f32[116]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_24: "f32[116]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_738: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_739: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    sum_50: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_80: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_740);  convolution_31 = unsqueeze_740 = None
    mul_360: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_80);  sub_80 = None
    sum_51: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 2, 3]);  mul_360 = None
    mul_365: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_95);  primals_95 = None
    unsqueeze_747: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_748: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_366: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_749);  where_16 = unsqueeze_749 = None
    mul_367: "f32[116]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_366, add_61, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_366 = add_61 = primals_94 = None
    getitem_100: "f32[4, 116, 14, 14]" = convolution_backward_24[0]
    getitem_101: "f32[116, 116, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    add_138: "f32[116]" = torch.ops.aten.add.Tensor(primals_262, 1e-05);  primals_262 = None
    rsqrt_25: "f32[116]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_750: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
    unsqueeze_751: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    sum_52: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_100, [0, 2, 3])
    sub_81: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_752);  convolution_30 = unsqueeze_752 = None
    mul_368: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_100, sub_81);  sub_81 = None
    sum_53: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3]);  mul_368 = None
    mul_373: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_92);  primals_92 = None
    unsqueeze_759: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
    unsqueeze_760: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_374: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_100, unsqueeze_761);  getitem_100 = unsqueeze_761 = None
    mul_375: "f32[116]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_374, relu_19, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_374 = primals_91 = None
    getitem_103: "f32[4, 116, 14, 14]" = convolution_backward_25[0]
    getitem_104: "f32[116, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    alias_89: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_90: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    le_17: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_90, 0);  alias_90 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_103);  le_17 = scalar_tensor_17 = getitem_103 = None
    add_139: "f32[116]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_26: "f32[116]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_762: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_763: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    sum_54: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_82: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_764);  convolution_29 = unsqueeze_764 = None
    mul_376: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_82);  sub_82 = None
    sum_55: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 2, 3]);  mul_376 = None
    mul_381: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_89);  primals_89 = None
    unsqueeze_771: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
    unsqueeze_772: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_382: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_773);  where_17 = unsqueeze_773 = None
    mul_383: "f32[116]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_382, getitem_15, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_382 = getitem_15 = primals_88 = None
    getitem_106: "f32[4, 116, 14, 14]" = convolution_backward_26[0]
    getitem_107: "f32[116, 116, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_22: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_15, getitem_106], 1);  slice_15 = getitem_106 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_49: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_22, [4, 116, 2, 14, 14]);  cat_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_29: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3, 4]);  view_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_24: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_50: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_24, [4, 232, 14, 14]);  clone_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_17: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_50, 1, 0, 116)
    slice_18: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_50, 1, 116, 232);  view_50 = None
    alias_92: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_93: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    le_18: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_93, 0);  alias_93 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, slice_18);  le_18 = scalar_tensor_18 = slice_18 = None
    add_140: "f32[116]" = torch.ops.aten.add.Tensor(primals_256, 1e-05);  primals_256 = None
    rsqrt_27: "f32[116]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_774: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_255, 0);  primals_255 = None
    unsqueeze_775: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    sum_56: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_83: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_776);  convolution_28 = unsqueeze_776 = None
    mul_384: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_83);  sub_83 = None
    sum_57: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3]);  mul_384 = None
    mul_389: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_86);  primals_86 = None
    unsqueeze_783: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_784: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_390: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_785);  where_18 = unsqueeze_785 = None
    mul_391: "f32[116]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_390, add_55, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_390 = add_55 = primals_85 = None
    getitem_109: "f32[4, 116, 14, 14]" = convolution_backward_27[0]
    getitem_110: "f32[116, 116, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    add_141: "f32[116]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_28: "f32[116]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_786: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_787: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    sum_58: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_109, [0, 2, 3])
    sub_84: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_788);  convolution_27 = unsqueeze_788 = None
    mul_392: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_109, sub_84);  sub_84 = None
    sum_59: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
    mul_397: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_83);  primals_83 = None
    unsqueeze_795: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
    unsqueeze_796: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_398: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_109, unsqueeze_797);  getitem_109 = unsqueeze_797 = None
    mul_399: "f32[116]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_398, relu_17, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_398 = primals_82 = None
    getitem_112: "f32[4, 116, 14, 14]" = convolution_backward_28[0]
    getitem_113: "f32[116, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    alias_95: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_96: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    le_19: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_96, 0);  alias_96 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_112);  le_19 = scalar_tensor_19 = getitem_112 = None
    add_142: "f32[116]" = torch.ops.aten.add.Tensor(primals_250, 1e-05);  primals_250 = None
    rsqrt_29: "f32[116]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    unsqueeze_798: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_249, 0);  primals_249 = None
    unsqueeze_799: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    sum_60: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_85: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_800);  convolution_26 = unsqueeze_800 = None
    mul_400: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_85);  sub_85 = None
    sum_61: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_405: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_80);  primals_80 = None
    unsqueeze_807: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_808: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_406: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_809);  where_19 = unsqueeze_809 = None
    mul_407: "f32[116]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_406, getitem_13, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = getitem_13 = primals_79 = None
    getitem_115: "f32[4, 116, 14, 14]" = convolution_backward_29[0]
    getitem_116: "f32[116, 116, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_23: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_17, getitem_115], 1);  slice_17 = getitem_115 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_51: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_23, [4, 116, 2, 14, 14]);  cat_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_30: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_25: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_52: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_25, [4, 232, 14, 14]);  clone_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_19: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_52, 1, 0, 116)
    slice_20: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_52, 1, 116, 232);  view_52 = None
    alias_98: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_99: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    le_20: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_99, 0);  alias_99 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, slice_20);  le_20 = scalar_tensor_20 = slice_20 = None
    add_143: "f32[116]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_30: "f32[116]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    unsqueeze_810: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_811: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    sum_62: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_86: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_812);  convolution_25 = unsqueeze_812 = None
    mul_408: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_86);  sub_86 = None
    sum_63: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
    mul_413: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_77);  primals_77 = None
    unsqueeze_819: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_820: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_414: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_821);  where_20 = unsqueeze_821 = None
    mul_415: "f32[116]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_414, add_49, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = add_49 = primals_76 = None
    getitem_118: "f32[4, 116, 14, 14]" = convolution_backward_30[0]
    getitem_119: "f32[116, 116, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    add_144: "f32[116]" = torch.ops.aten.add.Tensor(primals_244, 1e-05);  primals_244 = None
    rsqrt_31: "f32[116]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_822: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_243, 0);  primals_243 = None
    unsqueeze_823: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    sum_64: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_118, [0, 2, 3])
    sub_87: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_824);  convolution_24 = unsqueeze_824 = None
    mul_416: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_118, sub_87);  sub_87 = None
    sum_65: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 2, 3]);  mul_416 = None
    mul_421: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_74);  primals_74 = None
    unsqueeze_831: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_832: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_422: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_118, unsqueeze_833);  getitem_118 = unsqueeze_833 = None
    mul_423: "f32[116]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_422, relu_15, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_422 = primals_73 = None
    getitem_121: "f32[4, 116, 14, 14]" = convolution_backward_31[0]
    getitem_122: "f32[116, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    alias_101: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_102: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le_21: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_121);  le_21 = scalar_tensor_21 = getitem_121 = None
    add_145: "f32[116]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_32: "f32[116]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_834: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_835: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    sum_66: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_88: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_836);  convolution_23 = unsqueeze_836 = None
    mul_424: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_88);  sub_88 = None
    sum_67: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 2, 3]);  mul_424 = None
    mul_429: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_71);  primals_71 = None
    unsqueeze_843: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
    unsqueeze_844: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_430: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_845);  where_21 = unsqueeze_845 = None
    mul_431: "f32[116]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_430, getitem_11, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_430 = getitem_11 = primals_70 = None
    getitem_124: "f32[4, 116, 14, 14]" = convolution_backward_32[0]
    getitem_125: "f32[116, 116, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_24: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_19, getitem_124], 1);  slice_19 = getitem_124 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_53: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_24, [4, 116, 2, 14, 14]);  cat_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_31: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3, 4]);  view_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_26: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_54: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_26, [4, 232, 14, 14]);  clone_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_21: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_54, 1, 0, 116)
    slice_22: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_54, 1, 116, 232);  view_54 = None
    alias_104: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_105: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_22: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, slice_22);  le_22 = scalar_tensor_22 = slice_22 = None
    add_146: "f32[116]" = torch.ops.aten.add.Tensor(primals_238, 1e-05);  primals_238 = None
    rsqrt_33: "f32[116]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_846: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_237, 0);  primals_237 = None
    unsqueeze_847: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    sum_68: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_89: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_848);  convolution_22 = unsqueeze_848 = None
    mul_432: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_89);  sub_89 = None
    sum_69: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 2, 3]);  mul_432 = None
    mul_437: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_68);  primals_68 = None
    unsqueeze_855: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_856: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_438: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_857);  where_22 = unsqueeze_857 = None
    mul_439: "f32[116]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_438, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_438 = add_43 = primals_67 = None
    getitem_127: "f32[4, 116, 14, 14]" = convolution_backward_33[0]
    getitem_128: "f32[116, 116, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    add_147: "f32[116]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_34: "f32[116]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_858: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_859: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    sum_70: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_127, [0, 2, 3])
    sub_90: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_860);  convolution_21 = unsqueeze_860 = None
    mul_440: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_127, sub_90);  sub_90 = None
    sum_71: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 2, 3]);  mul_440 = None
    mul_445: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_65);  primals_65 = None
    unsqueeze_867: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
    unsqueeze_868: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_446: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_127, unsqueeze_869);  getitem_127 = unsqueeze_869 = None
    mul_447: "f32[116]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_446, relu_13, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_446 = primals_64 = None
    getitem_130: "f32[4, 116, 14, 14]" = convolution_backward_34[0]
    getitem_131: "f32[116, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    alias_107: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_108: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_23: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_130);  le_23 = scalar_tensor_23 = getitem_130 = None
    add_148: "f32[116]" = torch.ops.aten.add.Tensor(primals_232, 1e-05);  primals_232 = None
    rsqrt_35: "f32[116]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_870: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_231, 0);  primals_231 = None
    unsqueeze_871: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    sum_72: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_91: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_872);  convolution_20 = unsqueeze_872 = None
    mul_448: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_91);  sub_91 = None
    sum_73: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3]);  mul_448 = None
    mul_453: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_62);  primals_62 = None
    unsqueeze_879: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
    unsqueeze_880: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_454: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_881);  where_23 = unsqueeze_881 = None
    mul_455: "f32[116]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_454, getitem_9, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_454 = getitem_9 = primals_61 = None
    getitem_133: "f32[4, 116, 14, 14]" = convolution_backward_35[0]
    getitem_134: "f32[116, 116, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_25: "f32[4, 232, 14, 14]" = torch.ops.aten.cat.default([slice_21, getitem_133], 1);  slice_21 = getitem_133 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_55: "f32[4, 116, 2, 14, 14]" = torch.ops.aten.view.default(cat_25, [4, 116, 2, 14, 14]);  cat_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_32: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3, 4]);  view_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_27: "f32[4, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_56: "f32[4, 232, 14, 14]" = torch.ops.aten.view.default(clone_27, [4, 232, 14, 14]);  clone_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    slice_23: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_56, 1, 0, 116)
    slice_24: "f32[4, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_56, 1, 116, 232);  view_56 = None
    alias_110: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_111: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_24: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, slice_24);  le_24 = scalar_tensor_24 = slice_24 = None
    add_149: "f32[116]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_36: "f32[116]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_882: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_883: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    sum_74: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_92: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_884);  convolution_19 = unsqueeze_884 = None
    mul_456: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_92);  sub_92 = None
    sum_75: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 2, 3]);  mul_456 = None
    mul_461: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_59);  primals_59 = None
    unsqueeze_891: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_892: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_462: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_893);  where_24 = unsqueeze_893 = None
    mul_463: "f32[116]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_462, add_37, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_37 = primals_58 = None
    getitem_136: "f32[4, 116, 14, 14]" = convolution_backward_36[0]
    getitem_137: "f32[116, 116, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    add_150: "f32[116]" = torch.ops.aten.add.Tensor(primals_226, 1e-05);  primals_226 = None
    rsqrt_37: "f32[116]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    unsqueeze_894: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_225, 0);  primals_225 = None
    unsqueeze_895: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    sum_76: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_136, [0, 2, 3])
    sub_93: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_896);  convolution_18 = unsqueeze_896 = None
    mul_464: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_136, sub_93);  sub_93 = None
    sum_77: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_469: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_56);  primals_56 = None
    unsqueeze_903: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_904: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_470: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_136, unsqueeze_905);  getitem_136 = unsqueeze_905 = None
    mul_471: "f32[116]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_470, relu_11, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_470 = primals_55 = None
    getitem_139: "f32[4, 116, 28, 28]" = convolution_backward_37[0]
    getitem_140: "f32[116, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    alias_113: "f32[4, 116, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_114: "f32[4, 116, 28, 28]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_25: "b8[4, 116, 28, 28]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[4, 116, 28, 28]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_139);  le_25 = scalar_tensor_25 = getitem_139 = None
    add_151: "f32[116]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_38: "f32[116]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_906: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_907: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    sum_78: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_94: "f32[4, 116, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_908);  convolution_17 = unsqueeze_908 = None
    mul_472: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_94);  sub_94 = None
    sum_79: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_477: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_53);  primals_53 = None
    unsqueeze_915: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_916: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_478: "f32[4, 116, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_917);  where_25 = unsqueeze_917 = None
    mul_479: "f32[116]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_478, view_7, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_478 = primals_52 = None
    getitem_142: "f32[4, 116, 28, 28]" = convolution_backward_38[0]
    getitem_143: "f32[116, 116, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    alias_116: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_117: "f32[4, 116, 14, 14]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_26: "b8[4, 116, 14, 14]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[4, 116, 14, 14]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, slice_23);  le_26 = scalar_tensor_26 = slice_23 = None
    add_152: "f32[116]" = torch.ops.aten.add.Tensor(primals_220, 1e-05);  primals_220 = None
    rsqrt_39: "f32[116]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_918: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_219, 0);  primals_219 = None
    unsqueeze_919: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    sum_80: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_95: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_920);  convolution_16 = unsqueeze_920 = None
    mul_480: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_95);  sub_95 = None
    sum_81: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
    mul_485: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_50);  primals_50 = None
    unsqueeze_927: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_928: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_486: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_929);  where_26 = unsqueeze_929 = None
    mul_487: "f32[116]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_486, add_31, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = add_31 = primals_49 = None
    getitem_145: "f32[4, 116, 14, 14]" = convolution_backward_39[0]
    getitem_146: "f32[116, 116, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    add_153: "f32[116]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_40: "f32[116]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_930: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_931: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    sum_82: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_145, [0, 2, 3])
    sub_96: "f32[4, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_932);  convolution_15 = unsqueeze_932 = None
    mul_488: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_145, sub_96);  sub_96 = None
    sum_83: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 2, 3]);  mul_488 = None
    mul_493: "f32[116]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_47);  primals_47 = None
    unsqueeze_939: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
    unsqueeze_940: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_494: "f32[4, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_145, unsqueeze_941);  getitem_145 = unsqueeze_941 = None
    mul_495: "f32[116]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_494, view_7, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_494 = view_7 = primals_46 = None
    getitem_148: "f32[4, 116, 28, 28]" = convolution_backward_40[0]
    getitem_149: "f32[116, 1, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    add_154: "f32[4, 116, 28, 28]" = torch.ops.aten.add.Tensor(getitem_142, getitem_148);  getitem_142 = getitem_148 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_57: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.view.default(add_154, [4, 58, 2, 28, 28]);  add_154 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_33: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_28: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
    view_58: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_28, [4, 116, 28, 28]);  clone_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_25: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_58, 1, 0, 58)
    slice_26: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_58, 1, 58, 116);  view_58 = None
    alias_119: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_120: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_27: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, slice_26);  le_27 = scalar_tensor_27 = slice_26 = None
    add_155: "f32[58]" = torch.ops.aten.add.Tensor(primals_214, 1e-05);  primals_214 = None
    rsqrt_41: "f32[58]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_942: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_213, 0);  primals_213 = None
    unsqueeze_943: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    sum_84: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_97: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_944);  convolution_14 = unsqueeze_944 = None
    mul_496: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_97);  sub_97 = None
    sum_85: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 2, 3]);  mul_496 = None
    mul_501: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_44);  primals_44 = None
    unsqueeze_951: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_952: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_502: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_953);  where_27 = unsqueeze_953 = None
    mul_503: "f32[58]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_502, add_27, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_502 = add_27 = primals_43 = None
    getitem_151: "f32[4, 58, 28, 28]" = convolution_backward_41[0]
    getitem_152: "f32[58, 58, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    add_156: "f32[58]" = torch.ops.aten.add.Tensor(primals_211, 1e-05);  primals_211 = None
    rsqrt_42: "f32[58]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_954: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_210, 0);  primals_210 = None
    unsqueeze_955: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    sum_86: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_151, [0, 2, 3])
    sub_98: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_956);  convolution_13 = unsqueeze_956 = None
    mul_504: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_151, sub_98);  sub_98 = None
    sum_87: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 2, 3]);  mul_504 = None
    mul_509: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_41);  primals_41 = None
    unsqueeze_963: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
    unsqueeze_964: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_510: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_151, unsqueeze_965);  getitem_151 = unsqueeze_965 = None
    mul_511: "f32[58]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_510, relu_8, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_510 = primals_40 = None
    getitem_154: "f32[4, 58, 28, 28]" = convolution_backward_42[0]
    getitem_155: "f32[58, 1, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    alias_122: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_123: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le_28: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_154);  le_28 = scalar_tensor_28 = getitem_154 = None
    add_157: "f32[58]" = torch.ops.aten.add.Tensor(primals_208, 1e-05);  primals_208 = None
    rsqrt_43: "f32[58]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_966: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_207, 0);  primals_207 = None
    unsqueeze_967: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    sum_88: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_99: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_968);  convolution_12 = unsqueeze_968 = None
    mul_512: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_99);  sub_99 = None
    sum_89: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
    mul_517: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_38);  primals_38 = None
    unsqueeze_975: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_976: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_518: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_977);  where_28 = unsqueeze_977 = None
    mul_519: "f32[58]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_518, getitem_7, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_518 = getitem_7 = primals_37 = None
    getitem_157: "f32[4, 58, 28, 28]" = convolution_backward_43[0]
    getitem_158: "f32[58, 58, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_26: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([slice_25, getitem_157], 1);  slice_25 = getitem_157 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_59: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.view.default(cat_26, [4, 58, 2, 28, 28]);  cat_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_34: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3, 4]);  view_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_29: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    view_60: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_29, [4, 116, 28, 28]);  clone_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_27: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_60, 1, 0, 58)
    slice_28: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_60, 1, 58, 116);  view_60 = None
    alias_125: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_126: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    le_29: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, slice_28);  le_29 = scalar_tensor_29 = slice_28 = None
    add_158: "f32[58]" = torch.ops.aten.add.Tensor(primals_205, 1e-05);  primals_205 = None
    rsqrt_44: "f32[58]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    unsqueeze_978: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_204, 0);  primals_204 = None
    unsqueeze_979: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    sum_90: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_100: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_980);  convolution_11 = unsqueeze_980 = None
    mul_520: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_100);  sub_100 = None
    sum_91: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 2, 3]);  mul_520 = None
    mul_525: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_35);  primals_35 = None
    unsqueeze_987: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    unsqueeze_988: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_526: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_989);  where_29 = unsqueeze_989 = None
    mul_527: "f32[58]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_526, add_21, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = add_21 = primals_34 = None
    getitem_160: "f32[4, 58, 28, 28]" = convolution_backward_44[0]
    getitem_161: "f32[58, 58, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    add_159: "f32[58]" = torch.ops.aten.add.Tensor(primals_202, 1e-05);  primals_202 = None
    rsqrt_45: "f32[58]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_990: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_201, 0);  primals_201 = None
    unsqueeze_991: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    sum_92: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_160, [0, 2, 3])
    sub_101: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_992);  convolution_10 = unsqueeze_992 = None
    mul_528: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_160, sub_101);  sub_101 = None
    sum_93: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
    mul_533: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_32);  primals_32 = None
    unsqueeze_999: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_1000: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_534: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_160, unsqueeze_1001);  getitem_160 = unsqueeze_1001 = None
    mul_535: "f32[58]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_534, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_534 = primals_31 = None
    getitem_163: "f32[4, 58, 28, 28]" = convolution_backward_45[0]
    getitem_164: "f32[58, 1, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    alias_128: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_129: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    le_30: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_163);  le_30 = scalar_tensor_30 = getitem_163 = None
    add_160: "f32[58]" = torch.ops.aten.add.Tensor(primals_199, 1e-05);  primals_199 = None
    rsqrt_46: "f32[58]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    unsqueeze_1002: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_198, 0);  primals_198 = None
    unsqueeze_1003: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    sum_94: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_102: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1004);  convolution_9 = unsqueeze_1004 = None
    mul_536: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_102);  sub_102 = None
    sum_95: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
    mul_541: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_29);  primals_29 = None
    unsqueeze_1011: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_1012: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_542: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_1013);  where_30 = unsqueeze_1013 = None
    mul_543: "f32[58]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_542, getitem_5, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = getitem_5 = primals_28 = None
    getitem_166: "f32[4, 58, 28, 28]" = convolution_backward_46[0]
    getitem_167: "f32[58, 58, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_27: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([slice_27, getitem_166], 1);  slice_27 = getitem_166 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_61: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.view.default(cat_27, [4, 58, 2, 28, 28]);  cat_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_35: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3, 4]);  view_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_30: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_62: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_30, [4, 116, 28, 28]);  clone_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    slice_29: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_62, 1, 0, 58)
    slice_30: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_62, 1, 58, 116);  view_62 = None
    alias_131: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_132: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_31: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, slice_30);  le_31 = scalar_tensor_31 = slice_30 = None
    add_161: "f32[58]" = torch.ops.aten.add.Tensor(primals_196, 1e-05);  primals_196 = None
    rsqrt_47: "f32[58]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_1014: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_195, 0);  primals_195 = None
    unsqueeze_1015: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    sum_96: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_103: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1016);  convolution_8 = unsqueeze_1016 = None
    mul_544: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_103);  sub_103 = None
    sum_97: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
    mul_549: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_26);  primals_26 = None
    unsqueeze_1023: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_1024: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_550: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_1025);  where_31 = unsqueeze_1025 = None
    mul_551: "f32[58]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_550, add_15, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_550 = add_15 = primals_25 = None
    getitem_169: "f32[4, 58, 28, 28]" = convolution_backward_47[0]
    getitem_170: "f32[58, 58, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    add_162: "f32[58]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_48: "f32[58]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_1026: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_1027: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    sum_98: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_169, [0, 2, 3])
    sub_104: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1028);  convolution_7 = unsqueeze_1028 = None
    mul_552: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_169, sub_104);  sub_104 = None
    sum_99: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
    mul_557: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_23);  primals_23 = None
    unsqueeze_1035: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_1036: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_558: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_169, unsqueeze_1037);  getitem_169 = unsqueeze_1037 = None
    mul_559: "f32[58]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_558, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_558 = primals_22 = None
    getitem_172: "f32[4, 58, 28, 28]" = convolution_backward_48[0]
    getitem_173: "f32[58, 1, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    alias_134: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_135: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_32: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_172);  le_32 = scalar_tensor_32 = getitem_172 = None
    add_163: "f32[58]" = torch.ops.aten.add.Tensor(primals_190, 1e-05);  primals_190 = None
    rsqrt_49: "f32[58]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    unsqueeze_1038: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_189, 0);  primals_189 = None
    unsqueeze_1039: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    sum_100: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_105: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1040);  convolution_6 = unsqueeze_1040 = None
    mul_560: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_105);  sub_105 = None
    sum_101: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None
    mul_565: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_20);  primals_20 = None
    unsqueeze_1047: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_1048: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_566: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_1049);  where_32 = unsqueeze_1049 = None
    mul_567: "f32[58]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_566, getitem_3, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_566 = getitem_3 = primals_19 = None
    getitem_175: "f32[4, 58, 28, 28]" = convolution_backward_49[0]
    getitem_176: "f32[58, 58, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    cat_28: "f32[4, 116, 28, 28]" = torch.ops.aten.cat.default([slice_29, getitem_175], 1);  slice_29 = getitem_175 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    view_63: "f32[4, 58, 2, 28, 28]" = torch.ops.aten.view.default(cat_28, [4, 58, 2, 28, 28]);  cat_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    permute_36: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    clone_31: "f32[4, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_64: "f32[4, 116, 28, 28]" = torch.ops.aten.view.default(clone_31, [4, 116, 28, 28]);  clone_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    slice_31: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_64, 1, 0, 58)
    slice_32: "f32[4, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_64, 1, 58, 116);  view_64 = None
    alias_137: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_138: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    le_33: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_138, 0);  alias_138 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, slice_32);  le_33 = scalar_tensor_33 = slice_32 = None
    add_164: "f32[58]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_50: "f32[58]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_1050: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_1051: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    sum_102: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_106: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1052);  convolution_5 = unsqueeze_1052 = None
    mul_568: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_106);  sub_106 = None
    sum_103: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
    mul_573: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_17);  primals_17 = None
    unsqueeze_1059: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_1060: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_574: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_1061);  where_33 = unsqueeze_1061 = None
    mul_575: "f32[58]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_574, add_9, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = add_9 = primals_16 = None
    getitem_178: "f32[4, 58, 28, 28]" = convolution_backward_50[0]
    getitem_179: "f32[58, 58, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    add_165: "f32[58]" = torch.ops.aten.add.Tensor(primals_184, 1e-05);  primals_184 = None
    rsqrt_51: "f32[58]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    unsqueeze_1062: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_183, 0);  primals_183 = None
    unsqueeze_1063: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    sum_104: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_178, [0, 2, 3])
    sub_107: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1064);  convolution_4 = unsqueeze_1064 = None
    mul_576: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_178, sub_107);  sub_107 = None
    sum_105: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 2, 3]);  mul_576 = None
    mul_581: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_14);  primals_14 = None
    unsqueeze_1071: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_1072: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_582: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_178, unsqueeze_1073);  getitem_178 = unsqueeze_1073 = None
    mul_583: "f32[58]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_582, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_582 = primals_13 = None
    getitem_181: "f32[4, 58, 56, 56]" = convolution_backward_51[0]
    getitem_182: "f32[58, 1, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    alias_140: "f32[4, 58, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_141: "f32[4, 58, 56, 56]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    le_34: "b8[4, 58, 56, 56]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[4, 58, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_181);  le_34 = scalar_tensor_34 = getitem_181 = None
    add_166: "f32[58]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_52: "f32[58]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_1074: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_1075: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    sum_106: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_108: "f32[4, 58, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1076);  convolution_3 = unsqueeze_1076 = None
    mul_584: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_108);  sub_108 = None
    sum_107: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_589: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_11);  primals_11 = None
    unsqueeze_1083: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_1084: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_590: "f32[4, 58, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_1085);  where_34 = unsqueeze_1085 = None
    mul_591: "f32[58]" = torch.ops.aten.mul.Tensor(sum_107, rsqrt_52);  sum_107 = rsqrt_52 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_590, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_590 = primals_10 = None
    getitem_184: "f32[4, 24, 56, 56]" = convolution_backward_52[0]
    getitem_185: "f32[58, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    alias_143: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_144: "f32[4, 58, 28, 28]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_35: "b8[4, 58, 28, 28]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[4, 58, 28, 28]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, slice_31);  le_35 = scalar_tensor_35 = slice_31 = None
    add_167: "f32[58]" = torch.ops.aten.add.Tensor(primals_178, 1e-05);  primals_178 = None
    rsqrt_53: "f32[58]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_1086: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(primals_177, 0);  primals_177 = None
    unsqueeze_1087: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    sum_108: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_109: "f32[4, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1088);  convolution_2 = unsqueeze_1088 = None
    mul_592: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, sub_109);  sub_109 = None
    sum_109: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3]);  mul_592 = None
    mul_597: "f32[58]" = torch.ops.aten.mul.Tensor(rsqrt_53, primals_8);  primals_8 = None
    unsqueeze_1095: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_1096: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_598: "f32[4, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_1097);  where_35 = unsqueeze_1097 = None
    mul_599: "f32[58]" = torch.ops.aten.mul.Tensor(sum_109, rsqrt_53);  sum_109 = rsqrt_53 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_598, add_3, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_598 = add_3 = primals_7 = None
    getitem_187: "f32[4, 24, 28, 28]" = convolution_backward_53[0]
    getitem_188: "f32[58, 24, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    add_168: "f32[24]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_54: "f32[24]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    unsqueeze_1098: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_1099: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    sum_110: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_187, [0, 2, 3])
    sub_110: "f32[4, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1100);  convolution_1 = unsqueeze_1100 = None
    mul_600: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_187, sub_110);  sub_110 = None
    sum_111: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2, 3]);  mul_600 = None
    mul_605: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_54, primals_5);  primals_5 = None
    unsqueeze_1107: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_1108: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_606: "f32[4, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_187, unsqueeze_1109);  getitem_187 = unsqueeze_1109 = None
    mul_607: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, rsqrt_54);  sum_111 = rsqrt_54 = None
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_606, getitem, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_606 = getitem = primals_4 = None
    getitem_190: "f32[4, 24, 56, 56]" = convolution_backward_54[0]
    getitem_191: "f32[24, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    add_169: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_184, getitem_190);  getitem_184 = getitem_190 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:156, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 24, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_169, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_169 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:155, code: x = self.conv1(x)
    alias_146: "f32[4, 24, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_147: "f32[4, 24, 112, 112]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_36: "b8[4, 24, 112, 112]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[4, 24, 112, 112]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, max_pool2d_with_indices_backward);  le_36 = scalar_tensor_36 = max_pool2d_with_indices_backward = None
    add_170: "f32[24]" = torch.ops.aten.add.Tensor(primals_172, 1e-05);  primals_172 = None
    rsqrt_55: "f32[24]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_1110: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_171, 0);  primals_171 = None
    unsqueeze_1111: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    sum_112: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_111: "f32[4, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1112);  convolution = unsqueeze_1112 = None
    mul_608: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_36, sub_111);  sub_111 = None
    sum_113: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_613: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_55, primals_2);  primals_2 = None
    unsqueeze_1119: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_1120: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_614: "f32[4, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_1121);  where_36 = unsqueeze_1121 = None
    mul_615: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, rsqrt_55);  sum_113 = rsqrt_55 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_614, primals_339, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_614 = primals_339 = primals_1 = None
    getitem_194: "f32[24, 3, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    return pytree.tree_unflatten([addmm, getitem_194, mul_615, sum_112, getitem_191, mul_607, sum_110, getitem_188, mul_599, sum_108, getitem_185, mul_591, sum_106, getitem_182, mul_583, sum_104, getitem_179, mul_575, sum_102, getitem_176, mul_567, sum_100, getitem_173, mul_559, sum_98, getitem_170, mul_551, sum_96, getitem_167, mul_543, sum_94, getitem_164, mul_535, sum_92, getitem_161, mul_527, sum_90, getitem_158, mul_519, sum_88, getitem_155, mul_511, sum_86, getitem_152, mul_503, sum_84, getitem_149, mul_495, sum_82, getitem_146, mul_487, sum_80, getitem_143, mul_479, sum_78, getitem_140, mul_471, sum_76, getitem_137, mul_463, sum_74, getitem_134, mul_455, sum_72, getitem_131, mul_447, sum_70, getitem_128, mul_439, sum_68, getitem_125, mul_431, sum_66, getitem_122, mul_423, sum_64, getitem_119, mul_415, sum_62, getitem_116, mul_407, sum_60, getitem_113, mul_399, sum_58, getitem_110, mul_391, sum_56, getitem_107, mul_383, sum_54, getitem_104, mul_375, sum_52, getitem_101, mul_367, sum_50, getitem_98, mul_359, sum_48, getitem_95, mul_351, sum_46, getitem_92, mul_343, sum_44, getitem_89, mul_335, sum_42, getitem_86, mul_327, sum_40, getitem_83, mul_319, sum_38, getitem_80, mul_311, sum_36, getitem_77, mul_303, sum_34, getitem_74, mul_295, sum_32, getitem_71, mul_287, sum_30, getitem_68, mul_279, sum_28, getitem_65, mul_271, sum_26, getitem_62, mul_263, sum_24, getitem_59, mul_255, sum_22, getitem_56, mul_247, sum_20, getitem_53, mul_239, sum_18, getitem_50, mul_231, sum_16, getitem_47, mul_223, sum_14, getitem_44, mul_215, sum_12, getitem_41, mul_207, sum_10, getitem_38, mul_199, sum_8, getitem_35, mul_191, sum_6, getitem_32, mul_183, sum_4, getitem_29, mul_175, sum_2, permute_20, view_32, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    