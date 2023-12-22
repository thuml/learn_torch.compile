from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32]"; primals_2: "f32[32]"; primals_3: "f32[224]"; primals_4: "f32[224]"; primals_5: "f32[224]"; primals_6: "f32[224]"; primals_7: "f32[224]"; primals_8: "f32[224]"; primals_9: "f32[224]"; primals_10: "f32[224]"; primals_11: "f32[224]"; primals_12: "f32[224]"; primals_13: "f32[224]"; primals_14: "f32[224]"; primals_15: "f32[224]"; primals_16: "f32[224]"; primals_17: "f32[448]"; primals_18: "f32[448]"; primals_19: "f32[448]"; primals_20: "f32[448]"; primals_21: "f32[448]"; primals_22: "f32[448]"; primals_23: "f32[448]"; primals_24: "f32[448]"; primals_25: "f32[448]"; primals_26: "f32[448]"; primals_27: "f32[448]"; primals_28: "f32[448]"; primals_29: "f32[448]"; primals_30: "f32[448]"; primals_31: "f32[448]"; primals_32: "f32[448]"; primals_33: "f32[448]"; primals_34: "f32[448]"; primals_35: "f32[448]"; primals_36: "f32[448]"; primals_37: "f32[448]"; primals_38: "f32[448]"; primals_39: "f32[448]"; primals_40: "f32[448]"; primals_41: "f32[448]"; primals_42: "f32[448]"; primals_43: "f32[448]"; primals_44: "f32[448]"; primals_45: "f32[448]"; primals_46: "f32[448]"; primals_47: "f32[448]"; primals_48: "f32[448]"; primals_49: "f32[896]"; primals_50: "f32[896]"; primals_51: "f32[896]"; primals_52: "f32[896]"; primals_53: "f32[896]"; primals_54: "f32[896]"; primals_55: "f32[896]"; primals_56: "f32[896]"; primals_57: "f32[896]"; primals_58: "f32[896]"; primals_59: "f32[896]"; primals_60: "f32[896]"; primals_61: "f32[896]"; primals_62: "f32[896]"; primals_63: "f32[896]"; primals_64: "f32[896]"; primals_65: "f32[896]"; primals_66: "f32[896]"; primals_67: "f32[896]"; primals_68: "f32[896]"; primals_69: "f32[896]"; primals_70: "f32[896]"; primals_71: "f32[896]"; primals_72: "f32[896]"; primals_73: "f32[896]"; primals_74: "f32[896]"; primals_75: "f32[896]"; primals_76: "f32[896]"; primals_77: "f32[896]"; primals_78: "f32[896]"; primals_79: "f32[896]"; primals_80: "f32[896]"; primals_81: "f32[896]"; primals_82: "f32[896]"; primals_83: "f32[896]"; primals_84: "f32[896]"; primals_85: "f32[896]"; primals_86: "f32[896]"; primals_87: "f32[896]"; primals_88: "f32[896]"; primals_89: "f32[896]"; primals_90: "f32[896]"; primals_91: "f32[896]"; primals_92: "f32[896]"; primals_93: "f32[896]"; primals_94: "f32[896]"; primals_95: "f32[896]"; primals_96: "f32[896]"; primals_97: "f32[896]"; primals_98: "f32[896]"; primals_99: "f32[896]"; primals_100: "f32[896]"; primals_101: "f32[896]"; primals_102: "f32[896]"; primals_103: "f32[896]"; primals_104: "f32[896]"; primals_105: "f32[896]"; primals_106: "f32[896]"; primals_107: "f32[896]"; primals_108: "f32[896]"; primals_109: "f32[896]"; primals_110: "f32[896]"; primals_111: "f32[896]"; primals_112: "f32[896]"; primals_113: "f32[896]"; primals_114: "f32[896]"; primals_115: "f32[896]"; primals_116: "f32[896]"; primals_117: "f32[2240]"; primals_118: "f32[2240]"; primals_119: "f32[2240]"; primals_120: "f32[2240]"; primals_121: "f32[2240]"; primals_122: "f32[2240]"; primals_123: "f32[2240]"; primals_124: "f32[2240]"; primals_125: "f32[32, 3, 3, 3]"; primals_126: "f32[224, 32, 1, 1]"; primals_127: "f32[224, 112, 3, 3]"; primals_128: "f32[8, 224, 1, 1]"; primals_129: "f32[8]"; primals_130: "f32[224, 8, 1, 1]"; primals_131: "f32[224]"; primals_132: "f32[224, 224, 1, 1]"; primals_133: "f32[224, 32, 1, 1]"; primals_134: "f32[224, 224, 1, 1]"; primals_135: "f32[224, 112, 3, 3]"; primals_136: "f32[56, 224, 1, 1]"; primals_137: "f32[56]"; primals_138: "f32[224, 56, 1, 1]"; primals_139: "f32[224]"; primals_140: "f32[224, 224, 1, 1]"; primals_141: "f32[448, 224, 1, 1]"; primals_142: "f32[448, 112, 3, 3]"; primals_143: "f32[56, 448, 1, 1]"; primals_144: "f32[56]"; primals_145: "f32[448, 56, 1, 1]"; primals_146: "f32[448]"; primals_147: "f32[448, 448, 1, 1]"; primals_148: "f32[448, 224, 1, 1]"; primals_149: "f32[448, 448, 1, 1]"; primals_150: "f32[448, 112, 3, 3]"; primals_151: "f32[112, 448, 1, 1]"; primals_152: "f32[112]"; primals_153: "f32[448, 112, 1, 1]"; primals_154: "f32[448]"; primals_155: "f32[448, 448, 1, 1]"; primals_156: "f32[448, 448, 1, 1]"; primals_157: "f32[448, 112, 3, 3]"; primals_158: "f32[112, 448, 1, 1]"; primals_159: "f32[112]"; primals_160: "f32[448, 112, 1, 1]"; primals_161: "f32[448]"; primals_162: "f32[448, 448, 1, 1]"; primals_163: "f32[448, 448, 1, 1]"; primals_164: "f32[448, 112, 3, 3]"; primals_165: "f32[112, 448, 1, 1]"; primals_166: "f32[112]"; primals_167: "f32[448, 112, 1, 1]"; primals_168: "f32[448]"; primals_169: "f32[448, 448, 1, 1]"; primals_170: "f32[448, 448, 1, 1]"; primals_171: "f32[448, 112, 3, 3]"; primals_172: "f32[112, 448, 1, 1]"; primals_173: "f32[112]"; primals_174: "f32[448, 112, 1, 1]"; primals_175: "f32[448]"; primals_176: "f32[448, 448, 1, 1]"; primals_177: "f32[896, 448, 1, 1]"; primals_178: "f32[896, 112, 3, 3]"; primals_179: "f32[112, 896, 1, 1]"; primals_180: "f32[112]"; primals_181: "f32[896, 112, 1, 1]"; primals_182: "f32[896]"; primals_183: "f32[896, 896, 1, 1]"; primals_184: "f32[896, 448, 1, 1]"; primals_185: "f32[896, 896, 1, 1]"; primals_186: "f32[896, 112, 3, 3]"; primals_187: "f32[224, 896, 1, 1]"; primals_188: "f32[224]"; primals_189: "f32[896, 224, 1, 1]"; primals_190: "f32[896]"; primals_191: "f32[896, 896, 1, 1]"; primals_192: "f32[896, 896, 1, 1]"; primals_193: "f32[896, 112, 3, 3]"; primals_194: "f32[224, 896, 1, 1]"; primals_195: "f32[224]"; primals_196: "f32[896, 224, 1, 1]"; primals_197: "f32[896]"; primals_198: "f32[896, 896, 1, 1]"; primals_199: "f32[896, 896, 1, 1]"; primals_200: "f32[896, 112, 3, 3]"; primals_201: "f32[224, 896, 1, 1]"; primals_202: "f32[224]"; primals_203: "f32[896, 224, 1, 1]"; primals_204: "f32[896]"; primals_205: "f32[896, 896, 1, 1]"; primals_206: "f32[896, 896, 1, 1]"; primals_207: "f32[896, 112, 3, 3]"; primals_208: "f32[224, 896, 1, 1]"; primals_209: "f32[224]"; primals_210: "f32[896, 224, 1, 1]"; primals_211: "f32[896]"; primals_212: "f32[896, 896, 1, 1]"; primals_213: "f32[896, 896, 1, 1]"; primals_214: "f32[896, 112, 3, 3]"; primals_215: "f32[224, 896, 1, 1]"; primals_216: "f32[224]"; primals_217: "f32[896, 224, 1, 1]"; primals_218: "f32[896]"; primals_219: "f32[896, 896, 1, 1]"; primals_220: "f32[896, 896, 1, 1]"; primals_221: "f32[896, 112, 3, 3]"; primals_222: "f32[224, 896, 1, 1]"; primals_223: "f32[224]"; primals_224: "f32[896, 224, 1, 1]"; primals_225: "f32[896]"; primals_226: "f32[896, 896, 1, 1]"; primals_227: "f32[896, 896, 1, 1]"; primals_228: "f32[896, 112, 3, 3]"; primals_229: "f32[224, 896, 1, 1]"; primals_230: "f32[224]"; primals_231: "f32[896, 224, 1, 1]"; primals_232: "f32[896]"; primals_233: "f32[896, 896, 1, 1]"; primals_234: "f32[896, 896, 1, 1]"; primals_235: "f32[896, 112, 3, 3]"; primals_236: "f32[224, 896, 1, 1]"; primals_237: "f32[224]"; primals_238: "f32[896, 224, 1, 1]"; primals_239: "f32[896]"; primals_240: "f32[896, 896, 1, 1]"; primals_241: "f32[896, 896, 1, 1]"; primals_242: "f32[896, 112, 3, 3]"; primals_243: "f32[224, 896, 1, 1]"; primals_244: "f32[224]"; primals_245: "f32[896, 224, 1, 1]"; primals_246: "f32[896]"; primals_247: "f32[896, 896, 1, 1]"; primals_248: "f32[896, 896, 1, 1]"; primals_249: "f32[896, 112, 3, 3]"; primals_250: "f32[224, 896, 1, 1]"; primals_251: "f32[224]"; primals_252: "f32[896, 224, 1, 1]"; primals_253: "f32[896]"; primals_254: "f32[896, 896, 1, 1]"; primals_255: "f32[2240, 896, 1, 1]"; primals_256: "f32[2240, 112, 3, 3]"; primals_257: "f32[224, 2240, 1, 1]"; primals_258: "f32[224]"; primals_259: "f32[2240, 224, 1, 1]"; primals_260: "f32[2240]"; primals_261: "f32[2240, 2240, 1, 1]"; primals_262: "f32[2240, 896, 1, 1]"; primals_263: "f32[1000, 2240]"; primals_264: "f32[1000]"; primals_265: "f32[32]"; primals_266: "f32[32]"; primals_267: "f32[224]"; primals_268: "f32[224]"; primals_269: "f32[224]"; primals_270: "f32[224]"; primals_271: "f32[224]"; primals_272: "f32[224]"; primals_273: "f32[224]"; primals_274: "f32[224]"; primals_275: "f32[224]"; primals_276: "f32[224]"; primals_277: "f32[224]"; primals_278: "f32[224]"; primals_279: "f32[224]"; primals_280: "f32[224]"; primals_281: "f32[448]"; primals_282: "f32[448]"; primals_283: "f32[448]"; primals_284: "f32[448]"; primals_285: "f32[448]"; primals_286: "f32[448]"; primals_287: "f32[448]"; primals_288: "f32[448]"; primals_289: "f32[448]"; primals_290: "f32[448]"; primals_291: "f32[448]"; primals_292: "f32[448]"; primals_293: "f32[448]"; primals_294: "f32[448]"; primals_295: "f32[448]"; primals_296: "f32[448]"; primals_297: "f32[448]"; primals_298: "f32[448]"; primals_299: "f32[448]"; primals_300: "f32[448]"; primals_301: "f32[448]"; primals_302: "f32[448]"; primals_303: "f32[448]"; primals_304: "f32[448]"; primals_305: "f32[448]"; primals_306: "f32[448]"; primals_307: "f32[448]"; primals_308: "f32[448]"; primals_309: "f32[448]"; primals_310: "f32[448]"; primals_311: "f32[448]"; primals_312: "f32[448]"; primals_313: "f32[896]"; primals_314: "f32[896]"; primals_315: "f32[896]"; primals_316: "f32[896]"; primals_317: "f32[896]"; primals_318: "f32[896]"; primals_319: "f32[896]"; primals_320: "f32[896]"; primals_321: "f32[896]"; primals_322: "f32[896]"; primals_323: "f32[896]"; primals_324: "f32[896]"; primals_325: "f32[896]"; primals_326: "f32[896]"; primals_327: "f32[896]"; primals_328: "f32[896]"; primals_329: "f32[896]"; primals_330: "f32[896]"; primals_331: "f32[896]"; primals_332: "f32[896]"; primals_333: "f32[896]"; primals_334: "f32[896]"; primals_335: "f32[896]"; primals_336: "f32[896]"; primals_337: "f32[896]"; primals_338: "f32[896]"; primals_339: "f32[896]"; primals_340: "f32[896]"; primals_341: "f32[896]"; primals_342: "f32[896]"; primals_343: "f32[896]"; primals_344: "f32[896]"; primals_345: "f32[896]"; primals_346: "f32[896]"; primals_347: "f32[896]"; primals_348: "f32[896]"; primals_349: "f32[896]"; primals_350: "f32[896]"; primals_351: "f32[896]"; primals_352: "f32[896]"; primals_353: "f32[896]"; primals_354: "f32[896]"; primals_355: "f32[896]"; primals_356: "f32[896]"; primals_357: "f32[896]"; primals_358: "f32[896]"; primals_359: "f32[896]"; primals_360: "f32[896]"; primals_361: "f32[896]"; primals_362: "f32[896]"; primals_363: "f32[896]"; primals_364: "f32[896]"; primals_365: "f32[896]"; primals_366: "f32[896]"; primals_367: "f32[896]"; primals_368: "f32[896]"; primals_369: "f32[896]"; primals_370: "f32[896]"; primals_371: "f32[896]"; primals_372: "f32[896]"; primals_373: "f32[896]"; primals_374: "f32[896]"; primals_375: "f32[896]"; primals_376: "f32[896]"; primals_377: "f32[896]"; primals_378: "f32[896]"; primals_379: "f32[896]"; primals_380: "f32[896]"; primals_381: "f32[2240]"; primals_382: "f32[2240]"; primals_383: "f32[2240]"; primals_384: "f32[2240]"; primals_385: "f32[2240]"; primals_386: "f32[2240]"; primals_387: "f32[2240]"; primals_388: "f32[2240]"; primals_389: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_389, primals_125, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_266, torch.float32)
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
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[4, 224, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    convert_element_type_3: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    add_2: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[224]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_13: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_15: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 224, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[4, 224, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_127, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_269, torch.float32)
    convert_element_type_5: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    add_4: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[224]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_21: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_23: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[4, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_2, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_3: "f32[4, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[4, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_4: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_130, primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
    alias_4: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_9: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_2, sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_9, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    convert_element_type_7: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_272, torch.float32)
    add_6: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[224]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_10: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_27: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  unsqueeze_25 = None
    mul_11: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_29: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_12: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
    unsqueeze_30: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_31: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_133, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    convert_element_type_9: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    add_8: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[224]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_13: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_35: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_33);  unsqueeze_33 = None
    mul_14: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_37: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_15: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
    unsqueeze_38: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_39: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_10: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_7, add_9);  add_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_4: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    alias_5: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(relu_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_275, torch.float32)
    convert_element_type_11: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    add_11: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[224]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_16: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_43: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_41);  unsqueeze_41 = None
    mul_17: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_45: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_18: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_45);  mul_17 = unsqueeze_45 = None
    unsqueeze_46: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_47: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_47);  mul_18 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    convert_element_type_13: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_278, torch.float32)
    add_13: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[224]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_19: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_51: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_49);  unsqueeze_49 = None
    mul_20: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_53: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_21: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
    unsqueeze_54: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_55: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[4, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_136, primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[4, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_138, primals_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_9: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_22: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_6, sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_22, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    convert_element_type_15: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    add_15: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[224]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_23: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
    unsqueeze_59: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_57);  unsqueeze_57 = None
    mul_24: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_61: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_25: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_61);  mul_24 = unsqueeze_61 = None
    unsqueeze_62: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_63: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_63);  mul_25 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_17: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_16, relu_4);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_8: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    alias_10: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(relu_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[4, 448, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_281, torch.float32)
    convert_element_type_17: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    add_18: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[448]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_26: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_67: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_65);  unsqueeze_65 = None
    mul_27: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_69: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_28: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_69);  mul_27 = unsqueeze_69 = None
    unsqueeze_70: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_71: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 448, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_71);  mul_28 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[4, 448, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_142, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    convert_element_type_19: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_284, torch.float32)
    add_20: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[448]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_29: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_29, -1);  mul_29 = None
    unsqueeze_75: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_73);  unsqueeze_73 = None
    mul_30: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_77: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_31: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_77);  mul_30 = unsqueeze_77 = None
    unsqueeze_78: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_79: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_79);  mul_31 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_10, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_14: "f32[4, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_143, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[4, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_15: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_15);  convolution_15 = None
    alias_14: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_32: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_10, sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_32, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    convert_element_type_21: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    add_22: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[448]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_33: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_83: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_81);  unsqueeze_81 = None
    mul_34: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_85: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_35: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_85);  mul_34 = unsqueeze_85 = None
    unsqueeze_86: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_87: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_87);  mul_35 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_148, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_287, torch.float32)
    convert_element_type_23: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    add_24: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[448]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_11: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_36: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_91: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_89);  unsqueeze_89 = None
    mul_37: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_93: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_38: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_93);  mul_37 = unsqueeze_93 = None
    unsqueeze_94: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_95: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_25: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_95);  mul_38 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_26: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_23, add_25);  add_23 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_12: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    alias_15: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    convert_element_type_25: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_290, torch.float32)
    add_27: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[448]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_12: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_39: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_99: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_97);  unsqueeze_97 = None
    mul_40: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_101: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_41: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_101);  mul_40 = unsqueeze_101 = None
    unsqueeze_102: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_103: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_28: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_103);  mul_41 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_150, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    convert_element_type_27: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    add_29: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[448]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_13: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_42: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_107: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_105);  unsqueeze_105 = None
    mul_43: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_109: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_44: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_109);  mul_43 = unsqueeze_109 = None
    unsqueeze_110: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_111: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_30: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_111);  mul_44 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_14, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_20: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_151, primals_152, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_15: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_21: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None
    alias_19: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_45: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_14, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_45, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_293, torch.float32)
    convert_element_type_29: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    add_31: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[448]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_46: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_46, -1);  mul_46 = None
    unsqueeze_115: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_113);  unsqueeze_113 = None
    mul_47: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_117: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_48: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_117);  mul_47 = unsqueeze_117 = None
    unsqueeze_118: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_119: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_119);  mul_48 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_33: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_32, relu_12);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_16: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    alias_20: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    convert_element_type_31: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_296, torch.float32)
    add_34: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[448]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_15: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_49: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
    unsqueeze_123: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_121);  unsqueeze_121 = None
    mul_50: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_125: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_51: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_125);  mul_50 = unsqueeze_125 = None
    unsqueeze_126: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_127: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_35: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_127);  mul_51 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    convert_element_type_33: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    add_36: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[448]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_16: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_52: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_131: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_129);  unsqueeze_129 = None
    mul_53: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_133: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_54: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_133);  mul_53 = unsqueeze_133 = None
    unsqueeze_134: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_135: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_37: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_135);  mul_54 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_25: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_158, primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_19: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_26: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_19, primals_160, primals_161, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_26);  convolution_26 = None
    alias_24: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_55: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_18, sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_55, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_299, torch.float32)
    convert_element_type_35: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    add_38: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[448]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_17: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_56: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_139: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_137);  unsqueeze_137 = None
    mul_57: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_141: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_58: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_141);  mul_57 = unsqueeze_141 = None
    unsqueeze_142: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_143: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_39: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_143);  mul_58 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_40: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_39, relu_16);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_20: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    alias_25: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    convert_element_type_37: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_302, torch.float32)
    add_41: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[448]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_18: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_59: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_59, -1);  mul_59 = None
    unsqueeze_147: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_145);  unsqueeze_145 = None
    mul_60: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_149: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_61: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_60, unsqueeze_149);  mul_60 = unsqueeze_149 = None
    unsqueeze_150: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_151: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_42: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_61, unsqueeze_151);  mul_61 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_164, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    convert_element_type_39: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    add_43: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[448]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_19: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_62: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_155: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_153);  unsqueeze_153 = None
    mul_63: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_157: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_64: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_157);  mul_63 = unsqueeze_157 = None
    unsqueeze_158: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_159: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_44: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_159);  mul_64 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_30: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_23: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_31: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_167, primals_168, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31);  convolution_31 = None
    alias_29: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_65: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_22, sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_65, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_305, torch.float32)
    convert_element_type_41: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    add_45: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[448]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_20: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_66: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_163: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_161);  unsqueeze_161 = None
    mul_67: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_165: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_68: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_165);  mul_67 = unsqueeze_165 = None
    unsqueeze_166: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_167: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_46: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_167);  mul_68 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_47: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_46, relu_20);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_24: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    alias_30: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    convert_element_type_43: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_308, torch.float32)
    add_48: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[448]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_21: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_69: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_171: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_169);  unsqueeze_169 = None
    mul_70: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_173: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_71: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_173);  mul_70 = unsqueeze_173 = None
    unsqueeze_174: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_175: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_49: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_175);  mul_71 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    convert_element_type_45: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    add_50: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[448]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_22: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_72: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_179: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_177);  unsqueeze_177 = None
    mul_73: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_181: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_74: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_181);  mul_73 = unsqueeze_181 = None
    unsqueeze_182: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_183: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_51: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_183);  mul_74 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_35: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_172, primals_173, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_27: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_36: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_27, primals_174, primals_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_34: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_75: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_26, sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_37: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_75, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_311, torch.float32)
    convert_element_type_47: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_312, torch.float32)
    add_52: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[448]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_23: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_76: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_187: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_185);  unsqueeze_185 = None
    mul_77: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_189: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_78: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_189);  mul_77 = unsqueeze_189 = None
    unsqueeze_190: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_191: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_53: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_191);  mul_78 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_54: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_53, relu_24);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_28: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    alias_35: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[4, 896, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_313, torch.float32)
    convert_element_type_49: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_314, torch.float32)
    add_55: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[896]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_24: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_79: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_195: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_193);  unsqueeze_193 = None
    mul_80: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_197: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_81: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_197);  mul_80 = unsqueeze_197 = None
    unsqueeze_198: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_199: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_56: "f32[4, 896, 28, 28]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_199);  mul_81 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[4, 896, 28, 28]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_178, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_315, torch.float32)
    convert_element_type_51: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_316, torch.float32)
    add_57: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[896]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_25: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_82: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_203: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_201);  unsqueeze_201 = None
    mul_83: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_205: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_84: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_205);  mul_83 = unsqueeze_205 = None
    unsqueeze_206: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_207: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_58: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_207);  mul_84 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_40: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_179, primals_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_31: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_40);  convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_41: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_31, primals_181, primals_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_41);  convolution_41 = None
    alias_39: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_85: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_30, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_85, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_317, torch.float32)
    convert_element_type_53: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_318, torch.float32)
    add_59: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[896]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_26: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_86: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_86, -1);  mul_86 = None
    unsqueeze_211: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_209);  unsqueeze_209 = None
    mul_87: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_213: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_88: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_213);  mul_87 = unsqueeze_213 = None
    unsqueeze_214: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_215: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_60: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_215);  mul_88 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_184, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_319, torch.float32)
    convert_element_type_55: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_320, torch.float32)
    add_61: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[896]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_27: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_89: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_89, -1);  mul_89 = None
    unsqueeze_219: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_217);  unsqueeze_217 = None
    mul_90: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_221: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_91: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_221);  mul_90 = unsqueeze_221 = None
    unsqueeze_222: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_223: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_62: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_223);  mul_91 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_63: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_60, add_62);  add_60 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_32: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    alias_40: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_321, torch.float32)
    convert_element_type_57: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_322, torch.float32)
    add_64: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[896]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_28: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_92: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_227: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_225);  unsqueeze_225 = None
    mul_93: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_229: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_94: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_229);  mul_93 = unsqueeze_229 = None
    unsqueeze_230: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_231: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_65: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_231);  mul_94 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_186, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_323, torch.float32)
    convert_element_type_59: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_324, torch.float32)
    add_66: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[896]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_29: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_95: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_235: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_233);  unsqueeze_233 = None
    mul_96: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_237: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_97: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_237);  mul_96 = unsqueeze_237 = None
    unsqueeze_238: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_239: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_67: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_239);  mul_97 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_46: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_187, primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_35: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_47: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_35, primals_189, primals_190, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47);  convolution_47 = None
    alias_44: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_98: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_34, sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_98, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_325, torch.float32)
    convert_element_type_61: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_326, torch.float32)
    add_68: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[896]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_30: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_99: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_243: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_241);  unsqueeze_241 = None
    mul_100: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_245: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_101: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_245);  mul_100 = unsqueeze_245 = None
    unsqueeze_246: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_247: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_69: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_247);  mul_101 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_70: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_69, relu_32);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_36: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_70);  add_70 = None
    alias_45: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_327, torch.float32)
    convert_element_type_63: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_328, torch.float32)
    add_71: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[896]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_31: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_102: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_251: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_249);  unsqueeze_249 = None
    mul_103: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_253: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_104: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_253);  mul_103 = unsqueeze_253 = None
    unsqueeze_254: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_255: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_72: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_255);  mul_104 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_329, torch.float32)
    convert_element_type_65: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_330, torch.float32)
    add_73: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[896]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_32: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_105: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_259: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_257);  unsqueeze_257 = None
    mul_106: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_261: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_107: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_261);  mul_106 = unsqueeze_261 = None
    unsqueeze_262: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_263: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_74: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_263);  mul_107 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_51: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_39: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_51);  convolution_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_52: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    alias_49: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_108: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_38, sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_108, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_331, torch.float32)
    convert_element_type_67: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_332, torch.float32)
    add_75: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[896]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_33: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_109: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_109, -1);  mul_109 = None
    unsqueeze_267: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_265);  unsqueeze_265 = None
    mul_110: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_269: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_111: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_269);  mul_110 = unsqueeze_269 = None
    unsqueeze_270: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_271: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_76: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_271);  mul_111 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_77: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_76, relu_36);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_40: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    alias_50: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_199, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_333, torch.float32)
    convert_element_type_69: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_334, torch.float32)
    add_78: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[896]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_34: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_112: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
    unsqueeze_275: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_273);  unsqueeze_273 = None
    mul_113: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_277: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_114: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_277);  mul_113 = unsqueeze_277 = None
    unsqueeze_278: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_279: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_79: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_279);  mul_114 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_200, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_335, torch.float32)
    convert_element_type_71: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_336, torch.float32)
    add_80: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[896]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_35: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_115: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
    unsqueeze_283: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_281);  unsqueeze_281 = None
    mul_116: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_285: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_117: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_285);  mul_116 = unsqueeze_285 = None
    unsqueeze_286: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_287: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_81: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_287);  mul_117 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_56: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_43: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_57: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_43, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57);  convolution_57 = None
    alias_54: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_118: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_42, sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_118, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_337, torch.float32)
    convert_element_type_73: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_338, torch.float32)
    add_82: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[896]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_36: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_119: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
    unsqueeze_291: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_289);  unsqueeze_289 = None
    mul_120: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_293: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_121: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_293);  mul_120 = unsqueeze_293 = None
    unsqueeze_294: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_295: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_83: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_295);  mul_121 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_84: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_83, relu_40);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_44: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    alias_55: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_206, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_339, torch.float32)
    convert_element_type_75: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_340, torch.float32)
    add_85: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[896]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_37: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_122: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_122, -1);  mul_122 = None
    unsqueeze_299: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_297);  unsqueeze_297 = None
    mul_123: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_301: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_124: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_301);  mul_123 = unsqueeze_301 = None
    unsqueeze_302: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_303: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_86: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_303);  mul_124 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_207, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_341, torch.float32)
    convert_element_type_77: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_342, torch.float32)
    add_87: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[896]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_38: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_125: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_125, -1);  mul_125 = None
    unsqueeze_307: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_305);  unsqueeze_305 = None
    mul_126: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_309: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_127: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_309);  mul_126 = unsqueeze_309 = None
    unsqueeze_310: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_311: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_88: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_311);  mul_127 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_61: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_47: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_62: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_47, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
    alias_59: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_128: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_46, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_128, primals_212, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_78: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_343, torch.float32)
    convert_element_type_79: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_344, torch.float32)
    add_89: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[896]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_39: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_129: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_315: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_313);  unsqueeze_313 = None
    mul_130: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_317: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_131: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_317);  mul_130 = unsqueeze_317 = None
    unsqueeze_318: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_319: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_90: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_319);  mul_131 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_91: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_90, relu_44);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_48: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    alias_60: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_213, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_80: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_345, torch.float32)
    convert_element_type_81: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_346, torch.float32)
    add_92: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[896]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_40: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_132: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_323: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_321);  unsqueeze_321 = None
    mul_133: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_325: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_134: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_325);  mul_133 = unsqueeze_325 = None
    unsqueeze_326: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_327: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_93: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_327);  mul_134 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_214, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_82: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_347, torch.float32)
    convert_element_type_83: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_348, torch.float32)
    add_94: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[896]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_41: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_135: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_331: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_329);  unsqueeze_329 = None
    mul_136: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_333: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_137: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_333);  mul_136 = unsqueeze_333 = None
    unsqueeze_334: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_335: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_95: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_335);  mul_137 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_215, primals_216, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_51: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_51, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    alias_64: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_138: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_50, sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_68: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_138, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_84: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_349, torch.float32)
    convert_element_type_85: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_350, torch.float32)
    add_96: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[896]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_42: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_139: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
    unsqueeze_339: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_337);  unsqueeze_337 = None
    mul_140: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_341: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_141: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_341);  mul_140 = unsqueeze_341 = None
    unsqueeze_342: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_343: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_97: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_343);  mul_141 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_98: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_97, relu_48);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_52: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_98);  add_98 = None
    alias_65: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_86: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_351, torch.float32)
    convert_element_type_87: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_352, torch.float32)
    add_99: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[896]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_43: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_142: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
    unsqueeze_347: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_345);  unsqueeze_345 = None
    mul_143: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_349: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_144: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_349);  mul_143 = unsqueeze_349 = None
    unsqueeze_350: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_351: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_100: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_351);  mul_144 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_53: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_70: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_221, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_88: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_353, torch.float32)
    convert_element_type_89: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_354, torch.float32)
    add_101: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[896]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_44: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_145: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_145, -1);  mul_145 = None
    unsqueeze_355: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_353);  unsqueeze_353 = None
    mul_146: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_357: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_147: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_357);  mul_146 = unsqueeze_357 = None
    unsqueeze_358: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_359: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_102: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_359);  mul_147 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_54: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_54, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_71: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_222, primals_223, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_55: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_71);  convolution_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_72: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_55, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72);  convolution_72 = None
    alias_69: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_148: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_54, sigmoid_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_73: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_148, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_90: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_355, torch.float32)
    convert_element_type_91: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_356, torch.float32)
    add_103: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[896]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_45: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_149: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
    unsqueeze_363: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_361);  unsqueeze_361 = None
    mul_150: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_365: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_151: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_365);  mul_150 = unsqueeze_365 = None
    unsqueeze_366: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_367: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_104: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_367);  mul_151 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_105: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_104, relu_52);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_56: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    alias_70: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_74: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_92: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_357, torch.float32)
    convert_element_type_93: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_358, torch.float32)
    add_106: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[896]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_46: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_152: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_152, -1);  mul_152 = None
    unsqueeze_371: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_369);  unsqueeze_369 = None
    mul_153: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_373: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_154: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_373);  mul_153 = unsqueeze_373 = None
    unsqueeze_374: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_375: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_107: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_375);  mul_154 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_57: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_75: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_228, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_94: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_359, torch.float32)
    convert_element_type_95: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_360, torch.float32)
    add_108: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[896]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_47: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_155: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
    unsqueeze_379: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_377);  unsqueeze_377 = None
    mul_156: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_381: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_157: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_381);  mul_156 = unsqueeze_381 = None
    unsqueeze_382: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_383: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_109: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_383);  mul_157 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_58: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_76: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_229, primals_230, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_59: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_76);  convolution_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_77: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_59, primals_231, primals_232, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_14: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77);  convolution_77 = None
    alias_74: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_158: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_58, sigmoid_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_78: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_158, primals_233, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_96: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_361, torch.float32)
    convert_element_type_97: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_362, torch.float32)
    add_110: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[896]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_48: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_159: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_387: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_385);  unsqueeze_385 = None
    mul_160: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_389: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_161: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_389);  mul_160 = unsqueeze_389 = None
    unsqueeze_390: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_391: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_111: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_391);  mul_161 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_112: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_111, relu_56);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_60: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_112);  add_112 = None
    alias_75: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_79: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_98: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_363, torch.float32)
    convert_element_type_99: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_364, torch.float32)
    add_113: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[896]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
    reciprocal_49: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_162: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_395: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_393);  unsqueeze_393 = None
    mul_163: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_397: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_164: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_397);  mul_163 = unsqueeze_397 = None
    unsqueeze_398: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_399: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_114: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_399);  mul_164 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_61: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_80: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_235, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_100: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_365, torch.float32)
    convert_element_type_101: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_366, torch.float32)
    add_115: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[896]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_50: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_165: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_403: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_401);  unsqueeze_401 = None
    mul_166: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_405: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_167: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_405);  mul_166 = unsqueeze_405 = None
    unsqueeze_406: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_407: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_116: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_407);  mul_167 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_62: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_62, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_81: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_236, primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_63: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_81);  convolution_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_82: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_63, primals_238, primals_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_15: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82);  convolution_82 = None
    alias_79: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_168: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_62, sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_83: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_168, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_102: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_367, torch.float32)
    convert_element_type_103: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_368, torch.float32)
    add_117: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[896]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_51: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_169: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
    unsqueeze_411: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_409);  unsqueeze_409 = None
    mul_170: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_413: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_171: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_413);  mul_170 = unsqueeze_413 = None
    unsqueeze_414: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_415: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_118: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_415);  mul_171 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_119: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_118, relu_60);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_64: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    alias_80: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_84: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_104: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_369, torch.float32)
    convert_element_type_105: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_370, torch.float32)
    add_120: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[896]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_52: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_172: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
    unsqueeze_419: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_417);  unsqueeze_417 = None
    mul_173: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_421: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_174: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_421);  mul_173 = unsqueeze_421 = None
    unsqueeze_422: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_423: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_121: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_423);  mul_174 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_65: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_85: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_242, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_106: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_371, torch.float32)
    convert_element_type_107: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_372, torch.float32)
    add_122: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[896]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_53: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_175: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
    unsqueeze_427: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_425);  unsqueeze_425 = None
    mul_176: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_429: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_177: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_429);  mul_176 = unsqueeze_429 = None
    unsqueeze_430: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_431: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_123: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_431);  mul_177 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_66: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_123);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_16: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_66, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_86: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_243, primals_244, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_67: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_86);  convolution_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_87: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_67, primals_245, primals_246, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_16: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
    alias_84: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_178: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_66, sigmoid_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_88: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_178, primals_247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_108: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_373, torch.float32)
    convert_element_type_109: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_374, torch.float32)
    add_124: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[896]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_54: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_179: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
    unsqueeze_435: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_433);  unsqueeze_433 = None
    mul_180: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_437: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_181: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_437);  mul_180 = unsqueeze_437 = None
    unsqueeze_438: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_439: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_125: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_439);  mul_181 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_126: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_125, relu_64);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_68: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_126);  add_126 = None
    alias_85: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_89: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_248, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_110: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_375, torch.float32)
    convert_element_type_111: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_376, torch.float32)
    add_127: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[896]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_55: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_182: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_443: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_441);  unsqueeze_441 = None
    mul_183: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_445: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_184: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_445);  mul_183 = unsqueeze_445 = None
    unsqueeze_446: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_447: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_128: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_447);  mul_184 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_69: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_90: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_249, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_112: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_377, torch.float32)
    convert_element_type_113: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_378, torch.float32)
    add_129: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[896]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_56: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_185: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_185, -1);  mul_185 = None
    unsqueeze_451: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_449);  unsqueeze_449 = None
    mul_186: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_453: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_187: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_453);  mul_186 = unsqueeze_453 = None
    unsqueeze_454: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_455: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_130: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_455);  mul_187 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_70: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_17: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_70, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_91: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_250, primals_251, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_71: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_91);  convolution_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_92: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_71, primals_252, primals_253, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None
    alias_89: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_188: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_70, sigmoid_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_93: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_188, primals_254, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_114: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_379, torch.float32)
    convert_element_type_115: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_380, torch.float32)
    add_131: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[896]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_57: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_189: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_459: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_457);  unsqueeze_457 = None
    mul_190: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_461: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_191: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_461);  mul_190 = unsqueeze_461 = None
    unsqueeze_462: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_463: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_132: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_463);  mul_191 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_133: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_132, relu_68);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_72: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
    alias_90: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_94: "f32[4, 2240, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_255, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_116: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_381, torch.float32)
    convert_element_type_117: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_382, torch.float32)
    add_134: "f32[2240]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[2240]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_58: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_192: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_467: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[4, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_465);  unsqueeze_465 = None
    mul_193: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_469: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_194: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_469);  mul_193 = unsqueeze_469 = None
    unsqueeze_470: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_471: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_135: "f32[4, 2240, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_471);  mul_194 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_73: "f32[4, 2240, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_95: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_73, primals_256, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_118: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_383, torch.float32)
    convert_element_type_119: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_384, torch.float32)
    add_136: "f32[2240]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[2240]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_59: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_195: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_475: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_473);  unsqueeze_473 = None
    mul_196: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_477: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_197: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_477);  mul_196 = unsqueeze_477 = None
    unsqueeze_478: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_479: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_137: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_479);  mul_197 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_74: "f32[4, 2240, 7, 7]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_18: "f32[4, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_74, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_96: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_257, primals_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_75: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_96);  convolution_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_97: "f32[4, 2240, 1, 1]" = torch.ops.aten.convolution.default(relu_75, primals_259, primals_260, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18: "f32[4, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None
    alias_94: "f32[4, 2240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_198: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(relu_74, sigmoid_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_98: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(mul_198, primals_261, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_120: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_385, torch.float32)
    convert_element_type_121: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_386, torch.float32)
    add_138: "f32[2240]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[2240]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_60: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_199: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
    unsqueeze_483: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_481);  unsqueeze_481 = None
    mul_200: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_485: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_201: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_485);  mul_200 = unsqueeze_485 = None
    unsqueeze_486: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_487: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_139: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_487);  mul_201 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_99: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_72, primals_262, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_122: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_387, torch.float32)
    convert_element_type_123: "f32[2240]" = torch.ops.prims.convert_element_type.default(primals_388, torch.float32)
    add_140: "f32[2240]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[2240]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_61: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_202: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
    unsqueeze_491: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_489);  unsqueeze_489 = None
    mul_203: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_493: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_204: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_493);  mul_203 = unsqueeze_493 = None
    unsqueeze_494: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_495: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_141: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_495);  mul_204 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_142: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(add_139, add_141);  add_139 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_76: "f32[4, 2240, 7, 7]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    alias_95: "f32[4, 2240, 7, 7]" = torch.ops.aten.alias.default(relu_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_19: "f32[4, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_76, [-1, -2], True);  relu_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[4, 2240]" = torch.ops.aten.view.default(mean_19, [4, 2240]);  mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[4, 2240]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[2240, 1000]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_264, clone, permute);  primals_264 = None
    permute_1: "f32[1000, 2240]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 2240]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2240]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[2240, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2240]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[4, 2240, 1, 1]" = torch.ops.aten.view.default(mm, [4, 2240, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 2240, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 2240, 7, 7]);  view_2 = None
    div: "f32[4, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_96: "f32[4, 2240, 7, 7]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    le: "b8[4, 2240, 7, 7]" = torch.ops.aten.le.Scalar(alias_96, 0);  alias_96 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 2240, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_143: "f32[2240]" = torch.ops.aten.add.Tensor(primals_388, 1e-05);  primals_388 = None
    rsqrt: "f32[2240]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    unsqueeze_496: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_387, 0);  primals_387 = None
    unsqueeze_497: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_2: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_62: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_498);  convolution_99 = unsqueeze_498 = None
    mul_205: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_62);  sub_62 = None
    sum_3: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 2, 3]);  mul_205 = None
    mul_210: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt, primals_123);  primals_123 = None
    unsqueeze_505: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
    unsqueeze_506: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_211: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_507);  unsqueeze_507 = None
    mul_212: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_211, relu_72, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = primals_262 = None
    getitem: "f32[4, 896, 14, 14]" = convolution_backward[0]
    getitem_1: "f32[2240, 896, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_144: "f32[2240]" = torch.ops.aten.add.Tensor(primals_386, 1e-05);  primals_386 = None
    rsqrt_1: "f32[2240]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_508: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_385, 0);  primals_385 = None
    unsqueeze_509: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_4: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_63: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_510);  convolution_98 = unsqueeze_510 = None
    mul_213: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_63);  sub_63 = None
    sum_5: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3]);  mul_213 = None
    mul_218: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_121);  primals_121 = None
    unsqueeze_517: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
    unsqueeze_518: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_219: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_519);  where = unsqueeze_519 = None
    mul_220: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_219, mul_198, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = mul_198 = primals_261 = None
    getitem_3: "f32[4, 2240, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[2240, 2240, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_221: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, relu_74)
    mul_222: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid_18);  getitem_3 = sigmoid_18 = None
    sum_6: "f32[4, 2240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2, 3], True);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_97: "f32[4, 2240, 1, 1]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    sub_64: "f32[4, 2240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_97)
    mul_223: "f32[4, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_97, sub_64);  alias_97 = sub_64 = None
    mul_224: "f32[4, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_223);  sum_6 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_224, relu_75, primals_259, [2240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_224 = primals_259 = None
    getitem_6: "f32[4, 224, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[2240, 224, 1, 1]" = convolution_backward_2[1]
    getitem_8: "f32[2240]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_99: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_100: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_1: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_6);  le_1 = scalar_tensor_1 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_1, mean_18, primals_257, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = mean_18 = primals_257 = None
    getitem_9: "f32[4, 2240, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[224, 2240, 1, 1]" = convolution_backward_3[1]
    getitem_11: "f32[224]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 2240, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 2240, 7, 7]);  getitem_9 = None
    div_1: "f32[4, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_145: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_222, div_1);  mul_222 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_102: "f32[4, 2240, 7, 7]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_103: "f32[4, 2240, 7, 7]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_2: "b8[4, 2240, 7, 7]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 2240, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, add_145);  le_2 = scalar_tensor_2 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_146: "f32[2240]" = torch.ops.aten.add.Tensor(primals_384, 1e-05);  primals_384 = None
    rsqrt_2: "f32[2240]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_520: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_383, 0);  primals_383 = None
    unsqueeze_521: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_7: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_65: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_522);  convolution_95 = unsqueeze_522 = None
    mul_225: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_65);  sub_65 = None
    sum_8: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
    mul_230: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_119);  primals_119 = None
    unsqueeze_529: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_530: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_231: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_531);  where_2 = unsqueeze_531 = None
    mul_232: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_8, rsqrt_2);  sum_8 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_231, relu_73, primals_256, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_231 = primals_256 = None
    getitem_12: "f32[4, 2240, 14, 14]" = convolution_backward_4[0]
    getitem_13: "f32[2240, 112, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[4, 2240, 14, 14]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_106: "f32[4, 2240, 14, 14]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_3: "b8[4, 2240, 14, 14]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 2240, 14, 14]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_12);  le_3 = scalar_tensor_3 = getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_147: "f32[2240]" = torch.ops.aten.add.Tensor(primals_382, 1e-05);  primals_382 = None
    rsqrt_3: "f32[2240]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_532: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_381, 0);  primals_381 = None
    unsqueeze_533: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_9: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_66: "f32[4, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_534);  convolution_94 = unsqueeze_534 = None
    mul_233: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_66);  sub_66 = None
    sum_10: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 2, 3]);  mul_233 = None
    mul_238: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_117);  primals_117 = None
    unsqueeze_541: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_542: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_239: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_543);  where_3 = unsqueeze_543 = None
    mul_240: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_10, rsqrt_3);  sum_10 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_239, relu_72, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_239 = relu_72 = primals_255 = None
    getitem_15: "f32[4, 896, 14, 14]" = convolution_backward_5[0]
    getitem_16: "f32[2240, 896, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_148: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(getitem, getitem_15);  getitem = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_107: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_4: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, add_148);  le_4 = scalar_tensor_4 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_149: "f32[896]" = torch.ops.aten.add.Tensor(primals_380, 1e-05);  primals_380 = None
    rsqrt_4: "f32[896]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_544: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_379, 0);  primals_379 = None
    unsqueeze_545: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_11: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_67: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_546);  convolution_93 = unsqueeze_546 = None
    mul_241: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_67);  sub_67 = None
    sum_12: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_246: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_115);  primals_115 = None
    unsqueeze_553: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_554: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_247: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_555);  unsqueeze_555 = None
    mul_248: "f32[896]" = torch.ops.aten.mul.Tensor(sum_12, rsqrt_4);  sum_12 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_247, mul_188, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_247 = mul_188 = primals_254 = None
    getitem_18: "f32[4, 896, 14, 14]" = convolution_backward_6[0]
    getitem_19: "f32[896, 896, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_249: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_18, relu_70)
    mul_250: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_18, sigmoid_17);  getitem_18 = sigmoid_17 = None
    sum_13: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2, 3], True);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_108: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    sub_68: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_108)
    mul_251: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_108, sub_68);  alias_108 = sub_68 = None
    mul_252: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_251);  sum_13 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_252, relu_71, primals_252, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_252 = primals_252 = None
    getitem_21: "f32[4, 224, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[896, 224, 1, 1]" = convolution_backward_7[1]
    getitem_23: "f32[896]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_110: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_111: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_5: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_21);  le_5 = scalar_tensor_5 = getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_5, mean_17, primals_250, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = mean_17 = primals_250 = None
    getitem_24: "f32[4, 896, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[224, 896, 1, 1]" = convolution_backward_8[1]
    getitem_26: "f32[224]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_24, [4, 896, 14, 14]);  getitem_24 = None
    div_2: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_150: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_250, div_2);  mul_250 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_113: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_114: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_6: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_150);  le_6 = scalar_tensor_6 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_151: "f32[896]" = torch.ops.aten.add.Tensor(primals_378, 1e-05);  primals_378 = None
    rsqrt_5: "f32[896]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_556: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_377, 0);  primals_377 = None
    unsqueeze_557: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_14: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_69: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_558);  convolution_90 = unsqueeze_558 = None
    mul_253: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_69);  sub_69 = None
    sum_15: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 2, 3]);  mul_253 = None
    mul_258: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_113);  primals_113 = None
    unsqueeze_565: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
    unsqueeze_566: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_259: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_567);  where_6 = unsqueeze_567 = None
    mul_260: "f32[896]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_5);  sum_15 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_259, relu_69, primals_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_259 = primals_249 = None
    getitem_27: "f32[4, 896, 14, 14]" = convolution_backward_9[0]
    getitem_28: "f32[896, 112, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_116: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_117: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_7: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_27);  le_7 = scalar_tensor_7 = getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_152: "f32[896]" = torch.ops.aten.add.Tensor(primals_376, 1e-05);  primals_376 = None
    rsqrt_6: "f32[896]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_568: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_375, 0);  primals_375 = None
    unsqueeze_569: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_16: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_70: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_570);  convolution_89 = unsqueeze_570 = None
    mul_261: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_70);  sub_70 = None
    sum_17: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 2, 3]);  mul_261 = None
    mul_266: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_111);  primals_111 = None
    unsqueeze_577: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_266, 0);  mul_266 = None
    unsqueeze_578: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_267: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_579);  where_7 = unsqueeze_579 = None
    mul_268: "f32[896]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_6);  sum_17 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_267, relu_68, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_267 = relu_68 = primals_248 = None
    getitem_30: "f32[4, 896, 14, 14]" = convolution_backward_10[0]
    getitem_31: "f32[896, 896, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_153: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_4, getitem_30);  where_4 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_118: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_8: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_153);  le_8 = scalar_tensor_8 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_154: "f32[896]" = torch.ops.aten.add.Tensor(primals_374, 1e-05);  primals_374 = None
    rsqrt_7: "f32[896]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    unsqueeze_580: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_373, 0);  primals_373 = None
    unsqueeze_581: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_18: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_71: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_582);  convolution_88 = unsqueeze_582 = None
    mul_269: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_71);  sub_71 = None
    sum_19: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 2, 3]);  mul_269 = None
    mul_274: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_109);  primals_109 = None
    unsqueeze_589: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_590: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_275: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_591);  unsqueeze_591 = None
    mul_276: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_7);  sum_19 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_275, mul_178, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_275 = mul_178 = primals_247 = None
    getitem_33: "f32[4, 896, 14, 14]" = convolution_backward_11[0]
    getitem_34: "f32[896, 896, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_277: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_33, relu_66)
    mul_278: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_33, sigmoid_16);  getitem_33 = sigmoid_16 = None
    sum_20: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2, 3], True);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_119: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    sub_72: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_119)
    mul_279: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_119, sub_72);  alias_119 = sub_72 = None
    mul_280: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_279);  sum_20 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_280, relu_67, primals_245, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_280 = primals_245 = None
    getitem_36: "f32[4, 224, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[896, 224, 1, 1]" = convolution_backward_12[1]
    getitem_38: "f32[896]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_121: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_122: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    le_9: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_36);  le_9 = scalar_tensor_9 = getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_9, mean_16, primals_243, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_9 = mean_16 = primals_243 = None
    getitem_39: "f32[4, 896, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[224, 896, 1, 1]" = convolution_backward_13[1]
    getitem_41: "f32[224]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_39, [4, 896, 14, 14]);  getitem_39 = None
    div_3: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_3, 196);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_155: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, div_3);  mul_278 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_124: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_125: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    le_10: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_155);  le_10 = scalar_tensor_10 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_156: "f32[896]" = torch.ops.aten.add.Tensor(primals_372, 1e-05);  primals_372 = None
    rsqrt_8: "f32[896]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_592: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_371, 0);  primals_371 = None
    unsqueeze_593: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_21: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_73: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_594);  convolution_85 = unsqueeze_594 = None
    mul_281: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_73);  sub_73 = None
    sum_22: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3]);  mul_281 = None
    mul_286: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_107);  primals_107 = None
    unsqueeze_601: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_602: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_287: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_603);  where_10 = unsqueeze_603 = None
    mul_288: "f32[896]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_8);  sum_22 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_287, relu_65, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_287 = primals_242 = None
    getitem_42: "f32[4, 896, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[896, 112, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_127: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_128: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    le_11: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_128, 0);  alias_128 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_42);  le_11 = scalar_tensor_11 = getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_157: "f32[896]" = torch.ops.aten.add.Tensor(primals_370, 1e-05);  primals_370 = None
    rsqrt_9: "f32[896]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_604: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_369, 0);  primals_369 = None
    unsqueeze_605: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_23: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_74: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_606);  convolution_84 = unsqueeze_606 = None
    mul_289: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_74);  sub_74 = None
    sum_24: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_294: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_105);  primals_105 = None
    unsqueeze_613: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_614: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_295: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_615);  where_11 = unsqueeze_615 = None
    mul_296: "f32[896]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_9);  sum_24 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_295, relu_64, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_295 = relu_64 = primals_241 = None
    getitem_45: "f32[4, 896, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[896, 896, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_158: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_8, getitem_45);  where_8 = getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_129: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_12: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_158);  le_12 = scalar_tensor_12 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_159: "f32[896]" = torch.ops.aten.add.Tensor(primals_368, 1e-05);  primals_368 = None
    rsqrt_10: "f32[896]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_616: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_367, 0);  primals_367 = None
    unsqueeze_617: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_25: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_75: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_618);  convolution_83 = unsqueeze_618 = None
    mul_297: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_75);  sub_75 = None
    sum_26: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_302: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_103);  primals_103 = None
    unsqueeze_625: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_626: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_303: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_627);  unsqueeze_627 = None
    mul_304: "f32[896]" = torch.ops.aten.mul.Tensor(sum_26, rsqrt_10);  sum_26 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_303, mul_168, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = mul_168 = primals_240 = None
    getitem_48: "f32[4, 896, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[896, 896, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_305: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, relu_62)
    mul_306: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, sigmoid_15);  getitem_48 = sigmoid_15 = None
    sum_27: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2, 3], True);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_130: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    sub_76: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_130)
    mul_307: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_130, sub_76);  alias_130 = sub_76 = None
    mul_308: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, mul_307);  sum_27 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_308, relu_63, primals_238, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_308 = primals_238 = None
    getitem_51: "f32[4, 224, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[896, 224, 1, 1]" = convolution_backward_17[1]
    getitem_53: "f32[896]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_132: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_133: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_13: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_51);  le_13 = scalar_tensor_13 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_13, mean_15, primals_236, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_13 = mean_15 = primals_236 = None
    getitem_54: "f32[4, 896, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[224, 896, 1, 1]" = convolution_backward_18[1]
    getitem_56: "f32[224]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_54, [4, 896, 14, 14]);  getitem_54 = None
    div_4: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_160: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_306, div_4);  mul_306 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_135: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_136: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_14: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, add_160);  le_14 = scalar_tensor_14 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_161: "f32[896]" = torch.ops.aten.add.Tensor(primals_366, 1e-05);  primals_366 = None
    rsqrt_11: "f32[896]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_628: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_365, 0);  primals_365 = None
    unsqueeze_629: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_28: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_77: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_630);  convolution_80 = unsqueeze_630 = None
    mul_309: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_77);  sub_77 = None
    sum_29: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
    mul_314: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_101);  primals_101 = None
    unsqueeze_637: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_638: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_315: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_639);  where_14 = unsqueeze_639 = None
    mul_316: "f32[896]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_11);  sum_29 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_315, relu_61, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_315 = primals_235 = None
    getitem_57: "f32[4, 896, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[896, 112, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_138: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_139: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_15: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_57);  le_15 = scalar_tensor_15 = getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_162: "f32[896]" = torch.ops.aten.add.Tensor(primals_364, 1e-05);  primals_364 = None
    rsqrt_12: "f32[896]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_640: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_363, 0);  primals_363 = None
    unsqueeze_641: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_30: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_78: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_642);  convolution_79 = unsqueeze_642 = None
    mul_317: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_78);  sub_78 = None
    sum_31: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 2, 3]);  mul_317 = None
    mul_322: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_99);  primals_99 = None
    unsqueeze_649: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_650: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_323: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_651);  where_15 = unsqueeze_651 = None
    mul_324: "f32[896]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_12);  sum_31 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_323, relu_60, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_323 = relu_60 = primals_234 = None
    getitem_60: "f32[4, 896, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[896, 896, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_163: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_60);  where_12 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_140: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_16: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_140, 0);  alias_140 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, add_163);  le_16 = scalar_tensor_16 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_164: "f32[896]" = torch.ops.aten.add.Tensor(primals_362, 1e-05);  primals_362 = None
    rsqrt_13: "f32[896]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_652: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_361, 0);  primals_361 = None
    unsqueeze_653: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_32: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_79: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_654);  convolution_78 = unsqueeze_654 = None
    mul_325: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_79);  sub_79 = None
    sum_33: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3]);  mul_325 = None
    mul_330: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_97);  primals_97 = None
    unsqueeze_661: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_662: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_331: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_663);  unsqueeze_663 = None
    mul_332: "f32[896]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_13);  sum_33 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_331, mul_158, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_331 = mul_158 = primals_233 = None
    getitem_63: "f32[4, 896, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[896, 896, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_333: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, relu_58)
    mul_334: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_14);  getitem_63 = sigmoid_14 = None
    sum_34: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2, 3], True);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_141: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    sub_80: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_141)
    mul_335: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_141, sub_80);  alias_141 = sub_80 = None
    mul_336: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, mul_335);  sum_34 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_336, relu_59, primals_231, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_336 = primals_231 = None
    getitem_66: "f32[4, 224, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[896, 224, 1, 1]" = convolution_backward_22[1]
    getitem_68: "f32[896]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_143: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_144: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_17: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_66);  le_17 = scalar_tensor_17 = getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_17, mean_14, primals_229, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_17 = mean_14 = primals_229 = None
    getitem_69: "f32[4, 896, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[224, 896, 1, 1]" = convolution_backward_23[1]
    getitem_71: "f32[224]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_69, [4, 896, 14, 14]);  getitem_69 = None
    div_5: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_165: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_334, div_5);  mul_334 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_146: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_147: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_18: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_165);  le_18 = scalar_tensor_18 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_166: "f32[896]" = torch.ops.aten.add.Tensor(primals_360, 1e-05);  primals_360 = None
    rsqrt_14: "f32[896]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_664: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_359, 0);  primals_359 = None
    unsqueeze_665: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_35: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_81: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_666);  convolution_75 = unsqueeze_666 = None
    mul_337: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_81);  sub_81 = None
    sum_36: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_342: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_95);  primals_95 = None
    unsqueeze_673: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_674: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_343: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_675);  where_18 = unsqueeze_675 = None
    mul_344: "f32[896]" = torch.ops.aten.mul.Tensor(sum_36, rsqrt_14);  sum_36 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_343, relu_57, primals_228, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_343 = primals_228 = None
    getitem_72: "f32[4, 896, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[896, 112, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_149: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_150: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_19: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_72);  le_19 = scalar_tensor_19 = getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_167: "f32[896]" = torch.ops.aten.add.Tensor(primals_358, 1e-05);  primals_358 = None
    rsqrt_15: "f32[896]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_676: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_357, 0);  primals_357 = None
    unsqueeze_677: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_37: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_82: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_678);  convolution_74 = unsqueeze_678 = None
    mul_345: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_82);  sub_82 = None
    sum_38: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 2, 3]);  mul_345 = None
    mul_350: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_93);  primals_93 = None
    unsqueeze_685: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_686: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_351: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_687);  where_19 = unsqueeze_687 = None
    mul_352: "f32[896]" = torch.ops.aten.mul.Tensor(sum_38, rsqrt_15);  sum_38 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_351, relu_56, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_351 = relu_56 = primals_227 = None
    getitem_75: "f32[4, 896, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[896, 896, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_168: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_16, getitem_75);  where_16 = getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_151: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_20: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, add_168);  le_20 = scalar_tensor_20 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_169: "f32[896]" = torch.ops.aten.add.Tensor(primals_356, 1e-05);  primals_356 = None
    rsqrt_16: "f32[896]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_688: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_355, 0);  primals_355 = None
    unsqueeze_689: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_39: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_83: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_690);  convolution_73 = unsqueeze_690 = None
    mul_353: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_83);  sub_83 = None
    sum_40: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3]);  mul_353 = None
    mul_358: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_91);  primals_91 = None
    unsqueeze_697: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_698: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_359: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_699);  unsqueeze_699 = None
    mul_360: "f32[896]" = torch.ops.aten.mul.Tensor(sum_40, rsqrt_16);  sum_40 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_359, mul_148, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_148 = primals_226 = None
    getitem_78: "f32[4, 896, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[896, 896, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_361: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, relu_54)
    mul_362: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, sigmoid_13);  getitem_78 = sigmoid_13 = None
    sum_41: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_152: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    sub_84: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_152)
    mul_363: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_152, sub_84);  alias_152 = sub_84 = None
    mul_364: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, mul_363);  sum_41 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_364, relu_55, primals_224, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_364 = primals_224 = None
    getitem_81: "f32[4, 224, 1, 1]" = convolution_backward_27[0]
    getitem_82: "f32[896, 224, 1, 1]" = convolution_backward_27[1]
    getitem_83: "f32[896]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_154: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_155: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_154);  alias_154 = None
    le_21: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_155, 0);  alias_155 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_81);  le_21 = scalar_tensor_21 = getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_21, mean_13, primals_222, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_21 = mean_13 = primals_222 = None
    getitem_84: "f32[4, 896, 1, 1]" = convolution_backward_28[0]
    getitem_85: "f32[224, 896, 1, 1]" = convolution_backward_28[1]
    getitem_86: "f32[224]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_84, [4, 896, 14, 14]);  getitem_84 = None
    div_6: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_170: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, div_6);  mul_362 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_157: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_158: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_157);  alias_157 = None
    le_22: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_158, 0);  alias_158 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_170);  le_22 = scalar_tensor_22 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_171: "f32[896]" = torch.ops.aten.add.Tensor(primals_354, 1e-05);  primals_354 = None
    rsqrt_17: "f32[896]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_700: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_353, 0);  primals_353 = None
    unsqueeze_701: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_42: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_85: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_702);  convolution_70 = unsqueeze_702 = None
    mul_365: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_85);  sub_85 = None
    sum_43: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2, 3]);  mul_365 = None
    mul_370: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_89);  primals_89 = None
    unsqueeze_709: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_710: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_371: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_711);  where_22 = unsqueeze_711 = None
    mul_372: "f32[896]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_17);  sum_43 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_371, relu_53, primals_221, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_371 = primals_221 = None
    getitem_87: "f32[4, 896, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[896, 112, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_160: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_161: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_160);  alias_160 = None
    le_23: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_161, 0);  alias_161 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_87);  le_23 = scalar_tensor_23 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_172: "f32[896]" = torch.ops.aten.add.Tensor(primals_352, 1e-05);  primals_352 = None
    rsqrt_18: "f32[896]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_712: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_351, 0);  primals_351 = None
    unsqueeze_713: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_44: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_86: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_714);  convolution_69 = unsqueeze_714 = None
    mul_373: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_86);  sub_86 = None
    sum_45: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_378: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_87);  primals_87 = None
    unsqueeze_721: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_722: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_379: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_723);  where_23 = unsqueeze_723 = None
    mul_380: "f32[896]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_18);  sum_45 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_379, relu_52, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_379 = relu_52 = primals_220 = None
    getitem_90: "f32[4, 896, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[896, 896, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_173: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_20, getitem_90);  where_20 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_162: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_24: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, add_173);  le_24 = scalar_tensor_24 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_174: "f32[896]" = torch.ops.aten.add.Tensor(primals_350, 1e-05);  primals_350 = None
    rsqrt_19: "f32[896]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_724: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_349, 0);  primals_349 = None
    unsqueeze_725: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_46: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_87: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_726);  convolution_68 = unsqueeze_726 = None
    mul_381: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_87);  sub_87 = None
    sum_47: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
    mul_386: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_85);  primals_85 = None
    unsqueeze_733: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_734: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_387: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_735);  unsqueeze_735 = None
    mul_388: "f32[896]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_19);  sum_47 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_387, mul_138, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = mul_138 = primals_219 = None
    getitem_93: "f32[4, 896, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[896, 896, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_389: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, relu_50)
    mul_390: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, sigmoid_12);  getitem_93 = sigmoid_12 = None
    sum_48: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2, 3], True);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_163: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    sub_88: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_163)
    mul_391: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_163, sub_88);  alias_163 = sub_88 = None
    mul_392: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_391);  sum_48 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_392, relu_51, primals_217, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_392 = primals_217 = None
    getitem_96: "f32[4, 224, 1, 1]" = convolution_backward_32[0]
    getitem_97: "f32[896, 224, 1, 1]" = convolution_backward_32[1]
    getitem_98: "f32[896]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_165: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_166: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_165);  alias_165 = None
    le_25: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_96);  le_25 = scalar_tensor_25 = getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_25, mean_12, primals_215, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_25 = mean_12 = primals_215 = None
    getitem_99: "f32[4, 896, 1, 1]" = convolution_backward_33[0]
    getitem_100: "f32[224, 896, 1, 1]" = convolution_backward_33[1]
    getitem_101: "f32[224]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_99, [4, 896, 14, 14]);  getitem_99 = None
    div_7: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_175: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_390, div_7);  mul_390 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_168: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_169: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_26: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, add_175);  le_26 = scalar_tensor_26 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_176: "f32[896]" = torch.ops.aten.add.Tensor(primals_348, 1e-05);  primals_348 = None
    rsqrt_20: "f32[896]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    unsqueeze_736: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_347, 0);  primals_347 = None
    unsqueeze_737: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_49: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_89: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_738);  convolution_65 = unsqueeze_738 = None
    mul_393: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_89);  sub_89 = None
    sum_50: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2, 3]);  mul_393 = None
    mul_398: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_83);  primals_83 = None
    unsqueeze_745: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_746: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_399: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_747);  where_26 = unsqueeze_747 = None
    mul_400: "f32[896]" = torch.ops.aten.mul.Tensor(sum_50, rsqrt_20);  sum_50 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_399, relu_49, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_399 = primals_214 = None
    getitem_102: "f32[4, 896, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[896, 112, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_171: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_172: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_27: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_102);  le_27 = scalar_tensor_27 = getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_177: "f32[896]" = torch.ops.aten.add.Tensor(primals_346, 1e-05);  primals_346 = None
    rsqrt_21: "f32[896]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_748: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_345, 0);  primals_345 = None
    unsqueeze_749: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_51: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_90: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_750);  convolution_64 = unsqueeze_750 = None
    mul_401: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_90);  sub_90 = None
    sum_52: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 2, 3]);  mul_401 = None
    mul_406: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_81);  primals_81 = None
    unsqueeze_757: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
    unsqueeze_758: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_407: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_759);  where_27 = unsqueeze_759 = None
    mul_408: "f32[896]" = torch.ops.aten.mul.Tensor(sum_52, rsqrt_21);  sum_52 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_407, relu_48, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = relu_48 = primals_213 = None
    getitem_105: "f32[4, 896, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[896, 896, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_178: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_24, getitem_105);  where_24 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_173: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_28: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_173, 0);  alias_173 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, add_178);  le_28 = scalar_tensor_28 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_179: "f32[896]" = torch.ops.aten.add.Tensor(primals_344, 1e-05);  primals_344 = None
    rsqrt_22: "f32[896]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_760: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_343, 0);  primals_343 = None
    unsqueeze_761: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_53: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_91: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_762);  convolution_63 = unsqueeze_762 = None
    mul_409: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_91);  sub_91 = None
    sum_54: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_414: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_79);  primals_79 = None
    unsqueeze_769: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_770: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_415: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_771);  unsqueeze_771 = None
    mul_416: "f32[896]" = torch.ops.aten.mul.Tensor(sum_54, rsqrt_22);  sum_54 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_415, mul_128, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_415 = mul_128 = primals_212 = None
    getitem_108: "f32[4, 896, 14, 14]" = convolution_backward_36[0]
    getitem_109: "f32[896, 896, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_417: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, relu_46)
    mul_418: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, sigmoid_11);  getitem_108 = sigmoid_11 = None
    sum_55: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2, 3], True);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_174: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    sub_92: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_174)
    mul_419: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_174, sub_92);  alias_174 = sub_92 = None
    mul_420: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_419);  sum_55 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_420, relu_47, primals_210, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_420 = primals_210 = None
    getitem_111: "f32[4, 224, 1, 1]" = convolution_backward_37[0]
    getitem_112: "f32[896, 224, 1, 1]" = convolution_backward_37[1]
    getitem_113: "f32[896]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_176: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_177: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_29: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_111);  le_29 = scalar_tensor_29 = getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_29, mean_11, primals_208, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_29 = mean_11 = primals_208 = None
    getitem_114: "f32[4, 896, 1, 1]" = convolution_backward_38[0]
    getitem_115: "f32[224, 896, 1, 1]" = convolution_backward_38[1]
    getitem_116: "f32[224]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_114, [4, 896, 14, 14]);  getitem_114 = None
    div_8: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_180: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_418, div_8);  mul_418 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_179: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_180: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_30: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, add_180);  le_30 = scalar_tensor_30 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_181: "f32[896]" = torch.ops.aten.add.Tensor(primals_342, 1e-05);  primals_342 = None
    rsqrt_23: "f32[896]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_772: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_341, 0);  primals_341 = None
    unsqueeze_773: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_56: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_93: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_774);  convolution_60 = unsqueeze_774 = None
    mul_421: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_93);  sub_93 = None
    sum_57: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 2, 3]);  mul_421 = None
    mul_426: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_77);  primals_77 = None
    unsqueeze_781: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_782: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_427: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_783);  where_30 = unsqueeze_783 = None
    mul_428: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_23);  sum_57 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_427, relu_45, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_427 = primals_207 = None
    getitem_117: "f32[4, 896, 14, 14]" = convolution_backward_39[0]
    getitem_118: "f32[896, 112, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_182: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_183: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_31: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_117);  le_31 = scalar_tensor_31 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_182: "f32[896]" = torch.ops.aten.add.Tensor(primals_340, 1e-05);  primals_340 = None
    rsqrt_24: "f32[896]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_784: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_339, 0);  primals_339 = None
    unsqueeze_785: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_58: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_94: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_786);  convolution_59 = unsqueeze_786 = None
    mul_429: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_94);  sub_94 = None
    sum_59: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3]);  mul_429 = None
    mul_434: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_75);  primals_75 = None
    unsqueeze_793: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_794: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_435: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_795);  where_31 = unsqueeze_795 = None
    mul_436: "f32[896]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_24);  sum_59 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_435, relu_44, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_435 = relu_44 = primals_206 = None
    getitem_120: "f32[4, 896, 14, 14]" = convolution_backward_40[0]
    getitem_121: "f32[896, 896, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_183: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_28, getitem_120);  where_28 = getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_184: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_32: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_184, 0);  alias_184 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, add_183);  le_32 = scalar_tensor_32 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_184: "f32[896]" = torch.ops.aten.add.Tensor(primals_338, 1e-05);  primals_338 = None
    rsqrt_25: "f32[896]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    unsqueeze_796: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_337, 0);  primals_337 = None
    unsqueeze_797: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_60: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_95: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_798);  convolution_58 = unsqueeze_798 = None
    mul_437: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_95);  sub_95 = None
    sum_61: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_73);  primals_73 = None
    unsqueeze_805: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_806: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_443: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_807);  unsqueeze_807 = None
    mul_444: "f32[896]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_25);  sum_61 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_443, mul_118, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = mul_118 = primals_205 = None
    getitem_123: "f32[4, 896, 14, 14]" = convolution_backward_41[0]
    getitem_124: "f32[896, 896, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_445: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, relu_42)
    mul_446: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_10);  getitem_123 = sigmoid_10 = None
    sum_62: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2, 3], True);  mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_185: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    sub_96: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_185)
    mul_447: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_185, sub_96);  alias_185 = sub_96 = None
    mul_448: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_447);  sum_62 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_448, relu_43, primals_203, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_448 = primals_203 = None
    getitem_126: "f32[4, 224, 1, 1]" = convolution_backward_42[0]
    getitem_127: "f32[896, 224, 1, 1]" = convolution_backward_42[1]
    getitem_128: "f32[896]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_187: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_188: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_187);  alias_187 = None
    le_33: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_188, 0);  alias_188 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_126);  le_33 = scalar_tensor_33 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_33, mean_10, primals_201, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_33 = mean_10 = primals_201 = None
    getitem_129: "f32[4, 896, 1, 1]" = convolution_backward_43[0]
    getitem_130: "f32[224, 896, 1, 1]" = convolution_backward_43[1]
    getitem_131: "f32[224]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_129, [4, 896, 14, 14]);  getitem_129 = None
    div_9: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_185: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_446, div_9);  mul_446 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_190: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_191: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_190);  alias_190 = None
    le_34: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_191, 0);  alias_191 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, add_185);  le_34 = scalar_tensor_34 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_186: "f32[896]" = torch.ops.aten.add.Tensor(primals_336, 1e-05);  primals_336 = None
    rsqrt_26: "f32[896]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_808: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_335, 0);  primals_335 = None
    unsqueeze_809: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_63: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_97: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_810);  convolution_55 = unsqueeze_810 = None
    mul_449: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_97);  sub_97 = None
    sum_64: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
    mul_454: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_71);  primals_71 = None
    unsqueeze_817: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_818: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_455: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_819);  where_34 = unsqueeze_819 = None
    mul_456: "f32[896]" = torch.ops.aten.mul.Tensor(sum_64, rsqrt_26);  sum_64 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_455, relu_41, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_455 = primals_200 = None
    getitem_132: "f32[4, 896, 14, 14]" = convolution_backward_44[0]
    getitem_133: "f32[896, 112, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_193: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_194: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_193);  alias_193 = None
    le_35: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_194, 0);  alias_194 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, getitem_132);  le_35 = scalar_tensor_35 = getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_187: "f32[896]" = torch.ops.aten.add.Tensor(primals_334, 1e-05);  primals_334 = None
    rsqrt_27: "f32[896]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_820: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_333, 0);  primals_333 = None
    unsqueeze_821: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_65: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_98: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_822);  convolution_54 = unsqueeze_822 = None
    mul_457: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_98);  sub_98 = None
    sum_66: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
    mul_462: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_69);  primals_69 = None
    unsqueeze_829: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_830: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_463: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_831);  where_35 = unsqueeze_831 = None
    mul_464: "f32[896]" = torch.ops.aten.mul.Tensor(sum_66, rsqrt_27);  sum_66 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_463, relu_40, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_463 = relu_40 = primals_199 = None
    getitem_135: "f32[4, 896, 14, 14]" = convolution_backward_45[0]
    getitem_136: "f32[896, 896, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_188: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_32, getitem_135);  where_32 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_195: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_36: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_188);  le_36 = scalar_tensor_36 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_189: "f32[896]" = torch.ops.aten.add.Tensor(primals_332, 1e-05);  primals_332 = None
    rsqrt_28: "f32[896]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    unsqueeze_832: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_331, 0);  primals_331 = None
    unsqueeze_833: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_67: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_99: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_834);  convolution_53 = unsqueeze_834 = None
    mul_465: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_99);  sub_99 = None
    sum_68: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2, 3]);  mul_465 = None
    mul_470: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_67);  primals_67 = None
    unsqueeze_841: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
    unsqueeze_842: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_471: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_843);  unsqueeze_843 = None
    mul_472: "f32[896]" = torch.ops.aten.mul.Tensor(sum_68, rsqrt_28);  sum_68 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_471, mul_108, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_471 = mul_108 = primals_198 = None
    getitem_138: "f32[4, 896, 14, 14]" = convolution_backward_46[0]
    getitem_139: "f32[896, 896, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_473: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, relu_38)
    mul_474: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, sigmoid_9);  getitem_138 = sigmoid_9 = None
    sum_69: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2, 3], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_196: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    sub_100: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_196)
    mul_475: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_196, sub_100);  alias_196 = sub_100 = None
    mul_476: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_475);  sum_69 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_476, relu_39, primals_196, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_476 = primals_196 = None
    getitem_141: "f32[4, 224, 1, 1]" = convolution_backward_47[0]
    getitem_142: "f32[896, 224, 1, 1]" = convolution_backward_47[1]
    getitem_143: "f32[896]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_198: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_199: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_198);  alias_198 = None
    le_37: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_199, 0);  alias_199 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_141);  le_37 = scalar_tensor_37 = getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_37, mean_9, primals_194, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_37 = mean_9 = primals_194 = None
    getitem_144: "f32[4, 896, 1, 1]" = convolution_backward_48[0]
    getitem_145: "f32[224, 896, 1, 1]" = convolution_backward_48[1]
    getitem_146: "f32[224]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_144, [4, 896, 14, 14]);  getitem_144 = None
    div_10: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_190: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_474, div_10);  mul_474 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_201: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_202: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_201);  alias_201 = None
    le_38: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_202, 0);  alias_202 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, add_190);  le_38 = scalar_tensor_38 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_191: "f32[896]" = torch.ops.aten.add.Tensor(primals_330, 1e-05);  primals_330 = None
    rsqrt_29: "f32[896]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    unsqueeze_844: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_329, 0);  primals_329 = None
    unsqueeze_845: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_70: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_101: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_846);  convolution_50 = unsqueeze_846 = None
    mul_477: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_101);  sub_101 = None
    sum_71: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2, 3]);  mul_477 = None
    mul_482: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_65);  primals_65 = None
    unsqueeze_853: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_854: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_483: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_855);  where_38 = unsqueeze_855 = None
    mul_484: "f32[896]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_29);  sum_71 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_483, relu_37, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_483 = primals_193 = None
    getitem_147: "f32[4, 896, 14, 14]" = convolution_backward_49[0]
    getitem_148: "f32[896, 112, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_204: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_205: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_204);  alias_204 = None
    le_39: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_205, 0);  alias_205 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, getitem_147);  le_39 = scalar_tensor_39 = getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_192: "f32[896]" = torch.ops.aten.add.Tensor(primals_328, 1e-05);  primals_328 = None
    rsqrt_30: "f32[896]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_856: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_327, 0);  primals_327 = None
    unsqueeze_857: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_72: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_102: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_858);  convolution_49 = unsqueeze_858 = None
    mul_485: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_102);  sub_102 = None
    sum_73: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3]);  mul_485 = None
    mul_490: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_63);  primals_63 = None
    unsqueeze_865: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_866: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_491: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_867);  where_39 = unsqueeze_867 = None
    mul_492: "f32[896]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_30);  sum_73 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_491, relu_36, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_491 = relu_36 = primals_192 = None
    getitem_150: "f32[4, 896, 14, 14]" = convolution_backward_50[0]
    getitem_151: "f32[896, 896, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_193: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_150);  where_36 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_206: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_40: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_206, 0);  alias_206 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, add_193);  le_40 = scalar_tensor_40 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_194: "f32[896]" = torch.ops.aten.add.Tensor(primals_326, 1e-05);  primals_326 = None
    rsqrt_31: "f32[896]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    unsqueeze_868: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_325, 0);  primals_325 = None
    unsqueeze_869: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_74: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_103: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_870);  convolution_48 = unsqueeze_870 = None
    mul_493: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_103);  sub_103 = None
    sum_75: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 2, 3]);  mul_493 = None
    mul_498: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_61);  primals_61 = None
    unsqueeze_877: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    unsqueeze_878: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_499: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, unsqueeze_879);  unsqueeze_879 = None
    mul_500: "f32[896]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_31);  sum_75 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_499, mul_98, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_499 = mul_98 = primals_191 = None
    getitem_153: "f32[4, 896, 14, 14]" = convolution_backward_51[0]
    getitem_154: "f32[896, 896, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_501: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, relu_34)
    mul_502: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, sigmoid_8);  getitem_153 = sigmoid_8 = None
    sum_76: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2, 3], True);  mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_207: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    sub_104: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_207)
    mul_503: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_207, sub_104);  alias_207 = sub_104 = None
    mul_504: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, mul_503);  sum_76 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_504, relu_35, primals_189, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_504 = primals_189 = None
    getitem_156: "f32[4, 224, 1, 1]" = convolution_backward_52[0]
    getitem_157: "f32[896, 224, 1, 1]" = convolution_backward_52[1]
    getitem_158: "f32[896]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_209: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_210: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    le_41: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_156);  le_41 = scalar_tensor_41 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_41, mean_8, primals_187, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_41 = mean_8 = primals_187 = None
    getitem_159: "f32[4, 896, 1, 1]" = convolution_backward_53[0]
    getitem_160: "f32[224, 896, 1, 1]" = convolution_backward_53[1]
    getitem_161: "f32[224]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_159, [4, 896, 14, 14]);  getitem_159 = None
    div_11: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_195: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_502, div_11);  mul_502 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_212: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_213: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    le_42: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_213, 0);  alias_213 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, add_195);  le_42 = scalar_tensor_42 = add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_196: "f32[896]" = torch.ops.aten.add.Tensor(primals_324, 1e-05);  primals_324 = None
    rsqrt_32: "f32[896]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    unsqueeze_880: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_323, 0);  primals_323 = None
    unsqueeze_881: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_77: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_105: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_882);  convolution_45 = unsqueeze_882 = None
    mul_505: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_105);  sub_105 = None
    sum_78: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3]);  mul_505 = None
    mul_510: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_59);  primals_59 = None
    unsqueeze_889: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
    unsqueeze_890: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_511: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, unsqueeze_891);  where_42 = unsqueeze_891 = None
    mul_512: "f32[896]" = torch.ops.aten.mul.Tensor(sum_78, rsqrt_32);  sum_78 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_511, relu_33, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_511 = primals_186 = None
    getitem_162: "f32[4, 896, 14, 14]" = convolution_backward_54[0]
    getitem_163: "f32[896, 112, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_215: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_216: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    le_43: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, getitem_162);  le_43 = scalar_tensor_43 = getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_197: "f32[896]" = torch.ops.aten.add.Tensor(primals_322, 1e-05);  primals_322 = None
    rsqrt_33: "f32[896]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_892: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_321, 0);  primals_321 = None
    unsqueeze_893: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_79: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_106: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_894);  convolution_44 = unsqueeze_894 = None
    mul_513: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_106);  sub_106 = None
    sum_80: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3]);  mul_513 = None
    mul_518: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_57);  primals_57 = None
    unsqueeze_901: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_902: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_519: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_903);  where_43 = unsqueeze_903 = None
    mul_520: "f32[896]" = torch.ops.aten.mul.Tensor(sum_80, rsqrt_33);  sum_80 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_519, relu_32, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_519 = relu_32 = primals_185 = None
    getitem_165: "f32[4, 896, 14, 14]" = convolution_backward_55[0]
    getitem_166: "f32[896, 896, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_198: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_40, getitem_165);  where_40 = getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_217: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    le_44: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_217, 0);  alias_217 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, add_198);  le_44 = scalar_tensor_44 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_199: "f32[896]" = torch.ops.aten.add.Tensor(primals_320, 1e-05);  primals_320 = None
    rsqrt_34: "f32[896]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    unsqueeze_904: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_319, 0);  primals_319 = None
    unsqueeze_905: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_81: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_107: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_906);  convolution_43 = unsqueeze_906 = None
    mul_521: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_107);  sub_107 = None
    sum_82: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3]);  mul_521 = None
    mul_526: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_55);  primals_55 = None
    unsqueeze_913: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_914: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_527: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_915);  unsqueeze_915 = None
    mul_528: "f32[896]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_34);  sum_82 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_527, relu_28, primals_184, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_527 = primals_184 = None
    getitem_168: "f32[4, 448, 28, 28]" = convolution_backward_56[0]
    getitem_169: "f32[896, 448, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_200: "f32[896]" = torch.ops.aten.add.Tensor(primals_318, 1e-05);  primals_318 = None
    rsqrt_35: "f32[896]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    unsqueeze_916: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_317, 0);  primals_317 = None
    unsqueeze_917: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_83: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_108: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_918);  convolution_42 = unsqueeze_918 = None
    mul_529: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_108);  sub_108 = None
    sum_84: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2, 3]);  mul_529 = None
    mul_534: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_53);  primals_53 = None
    unsqueeze_925: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_926: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_535: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_927);  where_44 = unsqueeze_927 = None
    mul_536: "f32[896]" = torch.ops.aten.mul.Tensor(sum_84, rsqrt_35);  sum_84 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_535, mul_85, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = mul_85 = primals_183 = None
    getitem_171: "f32[4, 896, 14, 14]" = convolution_backward_57[0]
    getitem_172: "f32[896, 896, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_537: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_171, relu_30)
    mul_538: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_171, sigmoid_7);  getitem_171 = sigmoid_7 = None
    sum_85: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2, 3], True);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_218: "f32[4, 896, 1, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    sub_109: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_218)
    mul_539: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(alias_218, sub_109);  alias_218 = sub_109 = None
    mul_540: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_85, mul_539);  sum_85 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_540, relu_31, primals_181, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_540 = primals_181 = None
    getitem_174: "f32[4, 112, 1, 1]" = convolution_backward_58[0]
    getitem_175: "f32[896, 112, 1, 1]" = convolution_backward_58[1]
    getitem_176: "f32[896]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_220: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_221: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(alias_220);  alias_220 = None
    le_45: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(alias_221, 0);  alias_221 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, getitem_174);  le_45 = scalar_tensor_45 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_45, mean_7, primals_179, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_45 = mean_7 = primals_179 = None
    getitem_177: "f32[4, 896, 1, 1]" = convolution_backward_59[0]
    getitem_178: "f32[112, 896, 1, 1]" = convolution_backward_59[1]
    getitem_179: "f32[112]" = convolution_backward_59[2];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_177, [4, 896, 14, 14]);  getitem_177 = None
    div_12: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_201: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, div_12);  mul_538 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_223: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_224: "f32[4, 896, 14, 14]" = torch.ops.aten.alias.default(alias_223);  alias_223 = None
    le_46: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(alias_224, 0);  alias_224 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, add_201);  le_46 = scalar_tensor_46 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_202: "f32[896]" = torch.ops.aten.add.Tensor(primals_316, 1e-05);  primals_316 = None
    rsqrt_36: "f32[896]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    unsqueeze_928: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_315, 0);  primals_315 = None
    unsqueeze_929: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_86: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_110: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_930);  convolution_39 = unsqueeze_930 = None
    mul_541: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_110);  sub_110 = None
    sum_87: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 2, 3]);  mul_541 = None
    mul_546: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_51);  primals_51 = None
    unsqueeze_937: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_938: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_547: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, unsqueeze_939);  where_46 = unsqueeze_939 = None
    mul_548: "f32[896]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_36);  sum_87 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_547, relu_29, primals_178, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_547 = primals_178 = None
    getitem_180: "f32[4, 896, 28, 28]" = convolution_backward_60[0]
    getitem_181: "f32[896, 112, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_226: "f32[4, 896, 28, 28]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_227: "f32[4, 896, 28, 28]" = torch.ops.aten.alias.default(alias_226);  alias_226 = None
    le_47: "b8[4, 896, 28, 28]" = torch.ops.aten.le.Scalar(alias_227, 0);  alias_227 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[4, 896, 28, 28]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, getitem_180);  le_47 = scalar_tensor_47 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_203: "f32[896]" = torch.ops.aten.add.Tensor(primals_314, 1e-05);  primals_314 = None
    rsqrt_37: "f32[896]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    unsqueeze_940: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_313, 0);  primals_313 = None
    unsqueeze_941: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_88: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_111: "f32[4, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_942);  convolution_38 = unsqueeze_942 = None
    mul_549: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, sub_111);  sub_111 = None
    sum_89: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2, 3]);  mul_549 = None
    mul_554: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_49);  primals_49 = None
    unsqueeze_949: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_950: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_555: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_951);  where_47 = unsqueeze_951 = None
    mul_556: "f32[896]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_37);  sum_89 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_555, relu_28, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_555 = relu_28 = primals_177 = None
    getitem_183: "f32[4, 448, 28, 28]" = convolution_backward_61[0]
    getitem_184: "f32[896, 448, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_204: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(getitem_168, getitem_183);  getitem_168 = getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_228: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    le_48: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_228, 0);  alias_228 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, add_204);  le_48 = scalar_tensor_48 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_205: "f32[448]" = torch.ops.aten.add.Tensor(primals_312, 1e-05);  primals_312 = None
    rsqrt_38: "f32[448]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_952: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_311, 0);  primals_311 = None
    unsqueeze_953: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_90: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_112: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_954);  convolution_37 = unsqueeze_954 = None
    mul_557: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, sub_112);  sub_112 = None
    sum_91: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_562: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_47);  primals_47 = None
    unsqueeze_961: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_962: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_563: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, unsqueeze_963);  unsqueeze_963 = None
    mul_564: "f32[448]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_38);  sum_91 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_563, mul_75, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_563 = mul_75 = primals_176 = None
    getitem_186: "f32[4, 448, 28, 28]" = convolution_backward_62[0]
    getitem_187: "f32[448, 448, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_565: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_186, relu_26)
    mul_566: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_186, sigmoid_6);  getitem_186 = sigmoid_6 = None
    sum_92: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2, 3], True);  mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_229: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    sub_113: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_229)
    mul_567: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(alias_229, sub_113);  alias_229 = sub_113 = None
    mul_568: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_92, mul_567);  sum_92 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_568, relu_27, primals_174, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_568 = primals_174 = None
    getitem_189: "f32[4, 112, 1, 1]" = convolution_backward_63[0]
    getitem_190: "f32[448, 112, 1, 1]" = convolution_backward_63[1]
    getitem_191: "f32[448]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_231: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_232: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(alias_231);  alias_231 = None
    le_49: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(alias_232, 0);  alias_232 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_49, scalar_tensor_49, getitem_189);  le_49 = scalar_tensor_49 = getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_49, mean_6, primals_172, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_49 = mean_6 = primals_172 = None
    getitem_192: "f32[4, 448, 1, 1]" = convolution_backward_64[0]
    getitem_193: "f32[112, 448, 1, 1]" = convolution_backward_64[1]
    getitem_194: "f32[112]" = convolution_backward_64[2];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_192, [4, 448, 28, 28]);  getitem_192 = None
    div_13: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_206: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_566, div_13);  mul_566 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_234: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_235: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_234);  alias_234 = None
    le_50: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_235, 0);  alias_235 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_50: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_50, scalar_tensor_50, add_206);  le_50 = scalar_tensor_50 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_207: "f32[448]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_39: "f32[448]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    unsqueeze_964: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_965: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_93: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_114: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_966);  convolution_34 = unsqueeze_966 = None
    mul_569: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, sub_114);  sub_114 = None
    sum_94: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
    mul_574: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_45);  primals_45 = None
    unsqueeze_973: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
    unsqueeze_974: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_575: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, unsqueeze_975);  where_50 = unsqueeze_975 = None
    mul_576: "f32[448]" = torch.ops.aten.mul.Tensor(sum_94, rsqrt_39);  sum_94 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_575, relu_25, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_575 = primals_171 = None
    getitem_195: "f32[4, 448, 28, 28]" = convolution_backward_65[0]
    getitem_196: "f32[448, 112, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_237: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_238: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_237);  alias_237 = None
    le_51: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_238, 0);  alias_238 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_51: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_51, scalar_tensor_51, getitem_195);  le_51 = scalar_tensor_51 = getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_208: "f32[448]" = torch.ops.aten.add.Tensor(primals_308, 1e-05);  primals_308 = None
    rsqrt_40: "f32[448]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_976: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_307, 0);  primals_307 = None
    unsqueeze_977: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_95: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_115: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_978);  convolution_33 = unsqueeze_978 = None
    mul_577: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_115);  sub_115 = None
    sum_96: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3]);  mul_577 = None
    mul_582: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_43);  primals_43 = None
    unsqueeze_985: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_986: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_583: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, unsqueeze_987);  where_51 = unsqueeze_987 = None
    mul_584: "f32[448]" = torch.ops.aten.mul.Tensor(sum_96, rsqrt_40);  sum_96 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_583, relu_24, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_583 = relu_24 = primals_170 = None
    getitem_198: "f32[4, 448, 28, 28]" = convolution_backward_66[0]
    getitem_199: "f32[448, 448, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_209: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_48, getitem_198);  where_48 = getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_239: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_52: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_239, 0);  alias_239 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_52: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_52, scalar_tensor_52, add_209);  le_52 = scalar_tensor_52 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_210: "f32[448]" = torch.ops.aten.add.Tensor(primals_306, 1e-05);  primals_306 = None
    rsqrt_41: "f32[448]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    unsqueeze_988: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_305, 0);  primals_305 = None
    unsqueeze_989: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_97: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_116: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_990);  convolution_32 = unsqueeze_990 = None
    mul_585: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_116);  sub_116 = None
    sum_98: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3]);  mul_585 = None
    mul_590: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_41);  primals_41 = None
    unsqueeze_997: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_998: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_591: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, unsqueeze_999);  unsqueeze_999 = None
    mul_592: "f32[448]" = torch.ops.aten.mul.Tensor(sum_98, rsqrt_41);  sum_98 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_591, mul_65, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = mul_65 = primals_169 = None
    getitem_201: "f32[4, 448, 28, 28]" = convolution_backward_67[0]
    getitem_202: "f32[448, 448, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_593: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_201, relu_22)
    mul_594: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_201, sigmoid_5);  getitem_201 = sigmoid_5 = None
    sum_99: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2, 3], True);  mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_240: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    sub_117: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_240)
    mul_595: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(alias_240, sub_117);  alias_240 = sub_117 = None
    mul_596: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_99, mul_595);  sum_99 = mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_596, relu_23, primals_167, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_596 = primals_167 = None
    getitem_204: "f32[4, 112, 1, 1]" = convolution_backward_68[0]
    getitem_205: "f32[448, 112, 1, 1]" = convolution_backward_68[1]
    getitem_206: "f32[448]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_242: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_243: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    le_53: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_53: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_53, scalar_tensor_53, getitem_204);  le_53 = scalar_tensor_53 = getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(where_53, mean_5, primals_165, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_53 = mean_5 = primals_165 = None
    getitem_207: "f32[4, 448, 1, 1]" = convolution_backward_69[0]
    getitem_208: "f32[112, 448, 1, 1]" = convolution_backward_69[1]
    getitem_209: "f32[112]" = convolution_backward_69[2];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_207, [4, 448, 28, 28]);  getitem_207 = None
    div_14: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_211: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_594, div_14);  mul_594 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_245: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_246: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    le_54: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_246, 0);  alias_246 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_54: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_54, scalar_tensor_54, add_211);  le_54 = scalar_tensor_54 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_212: "f32[448]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_42: "f32[448]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_1000: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_1001: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_100: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_118: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1002);  convolution_29 = unsqueeze_1002 = None
    mul_597: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_118);  sub_118 = None
    sum_101: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_602: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_39);  primals_39 = None
    unsqueeze_1009: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_1010: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_603: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, unsqueeze_1011);  where_54 = unsqueeze_1011 = None
    mul_604: "f32[448]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_42);  sum_101 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_603, relu_21, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_603 = primals_164 = None
    getitem_210: "f32[4, 448, 28, 28]" = convolution_backward_70[0]
    getitem_211: "f32[448, 112, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_248: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_249: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    le_55: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_249, 0);  alias_249 = None
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_55: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_55, scalar_tensor_55, getitem_210);  le_55 = scalar_tensor_55 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_213: "f32[448]" = torch.ops.aten.add.Tensor(primals_302, 1e-05);  primals_302 = None
    rsqrt_43: "f32[448]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    unsqueeze_1012: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_301, 0);  primals_301 = None
    unsqueeze_1013: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sum_102: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_119: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1014);  convolution_28 = unsqueeze_1014 = None
    mul_605: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, sub_119);  sub_119 = None
    sum_103: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3]);  mul_605 = None
    mul_610: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_37);  primals_37 = None
    unsqueeze_1021: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_1022: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_611: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, unsqueeze_1023);  where_55 = unsqueeze_1023 = None
    mul_612: "f32[448]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_43);  sum_103 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_611, relu_20, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_611 = relu_20 = primals_163 = None
    getitem_213: "f32[4, 448, 28, 28]" = convolution_backward_71[0]
    getitem_214: "f32[448, 448, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_214: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_52, getitem_213);  where_52 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_250: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    le_56: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_250, 0);  alias_250 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_56: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_56, scalar_tensor_56, add_214);  le_56 = scalar_tensor_56 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_215: "f32[448]" = torch.ops.aten.add.Tensor(primals_300, 1e-05);  primals_300 = None
    rsqrt_44: "f32[448]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    unsqueeze_1024: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_299, 0);  primals_299 = None
    unsqueeze_1025: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_104: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_120: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1026);  convolution_27 = unsqueeze_1026 = None
    mul_613: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_120);  sub_120 = None
    sum_105: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 2, 3]);  mul_613 = None
    mul_618: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_35);  primals_35 = None
    unsqueeze_1033: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_1034: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_619: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, unsqueeze_1035);  unsqueeze_1035 = None
    mul_620: "f32[448]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_44);  sum_105 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_619, mul_55, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_619 = mul_55 = primals_162 = None
    getitem_216: "f32[4, 448, 28, 28]" = convolution_backward_72[0]
    getitem_217: "f32[448, 448, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_621: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_216, relu_18)
    mul_622: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_216, sigmoid_4);  getitem_216 = sigmoid_4 = None
    sum_106: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2, 3], True);  mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_251: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    sub_121: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_251)
    mul_623: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(alias_251, sub_121);  alias_251 = sub_121 = None
    mul_624: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_106, mul_623);  sum_106 = mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_624, relu_19, primals_160, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_624 = primals_160 = None
    getitem_219: "f32[4, 112, 1, 1]" = convolution_backward_73[0]
    getitem_220: "f32[448, 112, 1, 1]" = convolution_backward_73[1]
    getitem_221: "f32[448]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_253: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_254: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(alias_253);  alias_253 = None
    le_57: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(alias_254, 0);  alias_254 = None
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_57: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_57, scalar_tensor_57, getitem_219);  le_57 = scalar_tensor_57 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(where_57, mean_4, primals_158, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_57 = mean_4 = primals_158 = None
    getitem_222: "f32[4, 448, 1, 1]" = convolution_backward_74[0]
    getitem_223: "f32[112, 448, 1, 1]" = convolution_backward_74[1]
    getitem_224: "f32[112]" = convolution_backward_74[2];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_222, [4, 448, 28, 28]);  getitem_222 = None
    div_15: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_216: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_622, div_15);  mul_622 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_256: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_257: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_256);  alias_256 = None
    le_58: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_257, 0);  alias_257 = None
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_58: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_58, scalar_tensor_58, add_216);  le_58 = scalar_tensor_58 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_217: "f32[448]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_45: "f32[448]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    unsqueeze_1036: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_1037: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_107: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_122: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1038);  convolution_24 = unsqueeze_1038 = None
    mul_625: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, sub_122);  sub_122 = None
    sum_108: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
    mul_630: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_33);  primals_33 = None
    unsqueeze_1045: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_1046: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_631: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, unsqueeze_1047);  where_58 = unsqueeze_1047 = None
    mul_632: "f32[448]" = torch.ops.aten.mul.Tensor(sum_108, rsqrt_45);  sum_108 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_631, relu_17, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_631 = primals_157 = None
    getitem_225: "f32[4, 448, 28, 28]" = convolution_backward_75[0]
    getitem_226: "f32[448, 112, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_259: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_260: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_259);  alias_259 = None
    le_59: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_260, 0);  alias_260 = None
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_59: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_59, scalar_tensor_59, getitem_225);  le_59 = scalar_tensor_59 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_218: "f32[448]" = torch.ops.aten.add.Tensor(primals_296, 1e-05);  primals_296 = None
    rsqrt_46: "f32[448]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    unsqueeze_1048: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_295, 0);  primals_295 = None
    unsqueeze_1049: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_109: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_123: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1050);  convolution_23 = unsqueeze_1050 = None
    mul_633: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, sub_123);  sub_123 = None
    sum_110: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3]);  mul_633 = None
    mul_638: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_31);  primals_31 = None
    unsqueeze_1057: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_1058: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_639: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, unsqueeze_1059);  where_59 = unsqueeze_1059 = None
    mul_640: "f32[448]" = torch.ops.aten.mul.Tensor(sum_110, rsqrt_46);  sum_110 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_639, relu_16, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_639 = relu_16 = primals_156 = None
    getitem_228: "f32[4, 448, 28, 28]" = convolution_backward_76[0]
    getitem_229: "f32[448, 448, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_219: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_56, getitem_228);  where_56 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_261: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    le_60: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_261, 0);  alias_261 = None
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_60: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_60, scalar_tensor_60, add_219);  le_60 = scalar_tensor_60 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_220: "f32[448]" = torch.ops.aten.add.Tensor(primals_294, 1e-05);  primals_294 = None
    rsqrt_47: "f32[448]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    unsqueeze_1060: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_293, 0);  primals_293 = None
    unsqueeze_1061: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    sum_111: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_124: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1062);  convolution_22 = unsqueeze_1062 = None
    mul_641: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_124);  sub_124 = None
    sum_112: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2, 3]);  mul_641 = None
    mul_646: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_29);  primals_29 = None
    unsqueeze_1069: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_1070: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    mul_647: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, unsqueeze_1071);  unsqueeze_1071 = None
    mul_648: "f32[448]" = torch.ops.aten.mul.Tensor(sum_112, rsqrt_47);  sum_112 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_647, mul_45, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = mul_45 = primals_155 = None
    getitem_231: "f32[4, 448, 28, 28]" = convolution_backward_77[0]
    getitem_232: "f32[448, 448, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_649: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_231, relu_14)
    mul_650: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_231, sigmoid_3);  getitem_231 = sigmoid_3 = None
    sum_113: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2, 3], True);  mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_262: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_125: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_262)
    mul_651: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(alias_262, sub_125);  alias_262 = sub_125 = None
    mul_652: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_113, mul_651);  sum_113 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_652, relu_15, primals_153, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_652 = primals_153 = None
    getitem_234: "f32[4, 112, 1, 1]" = convolution_backward_78[0]
    getitem_235: "f32[448, 112, 1, 1]" = convolution_backward_78[1]
    getitem_236: "f32[448]" = convolution_backward_78[2];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_264: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_265: "f32[4, 112, 1, 1]" = torch.ops.aten.alias.default(alias_264);  alias_264 = None
    le_61: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(alias_265, 0);  alias_265 = None
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_61: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_61, scalar_tensor_61, getitem_234);  le_61 = scalar_tensor_61 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(where_61, mean_3, primals_151, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_61 = mean_3 = primals_151 = None
    getitem_237: "f32[4, 448, 1, 1]" = convolution_backward_79[0]
    getitem_238: "f32[112, 448, 1, 1]" = convolution_backward_79[1]
    getitem_239: "f32[112]" = convolution_backward_79[2];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_237, [4, 448, 28, 28]);  getitem_237 = None
    div_16: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_221: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_650, div_16);  mul_650 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_267: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_268: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_267);  alias_267 = None
    le_62: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_268, 0);  alias_268 = None
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_62: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_62, scalar_tensor_62, add_221);  le_62 = scalar_tensor_62 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_222: "f32[448]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_48: "f32[448]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    unsqueeze_1072: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_1073: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
    unsqueeze_1074: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None
    sum_114: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_126: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1074);  convolution_19 = unsqueeze_1074 = None
    mul_653: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_126);  sub_126 = None
    sum_115: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
    mul_658: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_27);  primals_27 = None
    unsqueeze_1081: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_1082: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
    unsqueeze_1083: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
    mul_659: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, unsqueeze_1083);  where_62 = unsqueeze_1083 = None
    mul_660: "f32[448]" = torch.ops.aten.mul.Tensor(sum_115, rsqrt_48);  sum_115 = rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_659, relu_13, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_659 = primals_150 = None
    getitem_240: "f32[4, 448, 28, 28]" = convolution_backward_80[0]
    getitem_241: "f32[448, 112, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_270: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_271: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_270);  alias_270 = None
    le_63: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_271, 0);  alias_271 = None
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_63: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_63, scalar_tensor_63, getitem_240);  le_63 = scalar_tensor_63 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_223: "f32[448]" = torch.ops.aten.add.Tensor(primals_290, 1e-05);  primals_290 = None
    rsqrt_49: "f32[448]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    unsqueeze_1084: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_289, 0);  primals_289 = None
    unsqueeze_1085: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
    unsqueeze_1086: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
    sum_116: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_127: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1086);  convolution_18 = unsqueeze_1086 = None
    mul_661: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_127);  sub_127 = None
    sum_117: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_666: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_25);  primals_25 = None
    unsqueeze_1093: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_1094: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
    unsqueeze_1095: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
    mul_667: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, unsqueeze_1095);  where_63 = unsqueeze_1095 = None
    mul_668: "f32[448]" = torch.ops.aten.mul.Tensor(sum_117, rsqrt_49);  sum_117 = rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_667, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = relu_12 = primals_149 = None
    getitem_243: "f32[4, 448, 28, 28]" = convolution_backward_81[0]
    getitem_244: "f32[448, 448, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_224: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_60, getitem_243);  where_60 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_272: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_64: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_272, 0);  alias_272 = None
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_64: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_64, scalar_tensor_64, add_224);  le_64 = scalar_tensor_64 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_225: "f32[448]" = torch.ops.aten.add.Tensor(primals_288, 1e-05);  primals_288 = None
    rsqrt_50: "f32[448]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    unsqueeze_1096: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_287, 0);  primals_287 = None
    unsqueeze_1097: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None
    sum_118: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_128: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1098);  convolution_17 = unsqueeze_1098 = None
    mul_669: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_128);  sub_128 = None
    sum_119: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3]);  mul_669 = None
    mul_674: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_23);  primals_23 = None
    unsqueeze_1105: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_1106: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
    unsqueeze_1107: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
    mul_675: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_1107);  unsqueeze_1107 = None
    mul_676: "f32[448]" = torch.ops.aten.mul.Tensor(sum_119, rsqrt_50);  sum_119 = rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_675, relu_8, primals_148, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = primals_148 = None
    getitem_246: "f32[4, 224, 56, 56]" = convolution_backward_82[0]
    getitem_247: "f32[448, 224, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_226: "f32[448]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_51: "f32[448]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    unsqueeze_1108: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_1109: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
    unsqueeze_1110: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None
    sum_120: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_129: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1110);  convolution_16 = unsqueeze_1110 = None
    mul_677: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_129);  sub_129 = None
    sum_121: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
    mul_682: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_21);  primals_21 = None
    unsqueeze_1117: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_1118: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
    unsqueeze_1119: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
    mul_683: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_1119);  where_64 = unsqueeze_1119 = None
    mul_684: "f32[448]" = torch.ops.aten.mul.Tensor(sum_121, rsqrt_51);  sum_121 = rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_683, mul_32, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_683 = mul_32 = primals_147 = None
    getitem_249: "f32[4, 448, 28, 28]" = convolution_backward_83[0]
    getitem_250: "f32[448, 448, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_685: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_249, relu_10)
    mul_686: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_249, sigmoid_2);  getitem_249 = sigmoid_2 = None
    sum_122: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_685, [2, 3], True);  mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_273: "f32[4, 448, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_130: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_273)
    mul_687: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(alias_273, sub_130);  alias_273 = sub_130 = None
    mul_688: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_122, mul_687);  sum_122 = mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_688, relu_11, primals_145, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_688 = primals_145 = None
    getitem_252: "f32[4, 56, 1, 1]" = convolution_backward_84[0]
    getitem_253: "f32[448, 56, 1, 1]" = convolution_backward_84[1]
    getitem_254: "f32[448]" = convolution_backward_84[2];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_275: "f32[4, 56, 1, 1]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_276: "f32[4, 56, 1, 1]" = torch.ops.aten.alias.default(alias_275);  alias_275 = None
    le_65: "b8[4, 56, 1, 1]" = torch.ops.aten.le.Scalar(alias_276, 0);  alias_276 = None
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_65: "f32[4, 56, 1, 1]" = torch.ops.aten.where.self(le_65, scalar_tensor_65, getitem_252);  le_65 = scalar_tensor_65 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(where_65, mean_2, primals_143, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_65 = mean_2 = primals_143 = None
    getitem_255: "f32[4, 448, 1, 1]" = convolution_backward_85[0]
    getitem_256: "f32[56, 448, 1, 1]" = convolution_backward_85[1]
    getitem_257: "f32[56]" = convolution_backward_85[2];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_17: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_255, [4, 448, 28, 28]);  getitem_255 = None
    div_17: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_17, 784);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_227: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_686, div_17);  mul_686 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_278: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_279: "f32[4, 448, 28, 28]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
    le_66: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_66: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_66, scalar_tensor_66, add_227);  le_66 = scalar_tensor_66 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_228: "f32[448]" = torch.ops.aten.add.Tensor(primals_284, 1e-05);  primals_284 = None
    rsqrt_52: "f32[448]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    unsqueeze_1120: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_283, 0);  primals_283 = None
    unsqueeze_1121: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
    unsqueeze_1122: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None
    sum_123: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_131: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1122);  convolution_13 = unsqueeze_1122 = None
    mul_689: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, sub_131);  sub_131 = None
    sum_124: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_694: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_19);  primals_19 = None
    unsqueeze_1129: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_1130: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
    unsqueeze_1131: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
    mul_695: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, unsqueeze_1131);  where_66 = unsqueeze_1131 = None
    mul_696: "f32[448]" = torch.ops.aten.mul.Tensor(sum_124, rsqrt_52);  sum_124 = rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_695, relu_9, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_695 = primals_142 = None
    getitem_258: "f32[4, 448, 56, 56]" = convolution_backward_86[0]
    getitem_259: "f32[448, 112, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_281: "f32[4, 448, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_282: "f32[4, 448, 56, 56]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_67: "b8[4, 448, 56, 56]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_67: "f32[4, 448, 56, 56]" = torch.ops.aten.where.self(le_67, scalar_tensor_67, getitem_258);  le_67 = scalar_tensor_67 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_229: "f32[448]" = torch.ops.aten.add.Tensor(primals_282, 1e-05);  primals_282 = None
    rsqrt_53: "f32[448]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    unsqueeze_1132: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_281, 0);  primals_281 = None
    unsqueeze_1133: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
    unsqueeze_1134: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None
    sum_125: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_132: "f32[4, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1134);  convolution_12 = unsqueeze_1134 = None
    mul_697: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, sub_132);  sub_132 = None
    sum_126: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_702: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_53, primals_17);  primals_17 = None
    unsqueeze_1141: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_1142: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
    unsqueeze_1143: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
    mul_703: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, unsqueeze_1143);  where_67 = unsqueeze_1143 = None
    mul_704: "f32[448]" = torch.ops.aten.mul.Tensor(sum_126, rsqrt_53);  sum_126 = rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_703, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = relu_8 = primals_141 = None
    getitem_261: "f32[4, 224, 56, 56]" = convolution_backward_87[0]
    getitem_262: "f32[448, 224, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_230: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(getitem_246, getitem_261);  getitem_246 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_283: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    le_68: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(alias_283, 0);  alias_283 = None
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_68: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_68, scalar_tensor_68, add_230);  le_68 = scalar_tensor_68 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_231: "f32[224]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_54: "f32[224]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    unsqueeze_1144: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_1145: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
    sum_127: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_133: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1146);  convolution_11 = unsqueeze_1146 = None
    mul_705: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, sub_133);  sub_133 = None
    sum_128: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
    mul_710: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_54, primals_15);  primals_15 = None
    unsqueeze_1153: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_1154: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
    unsqueeze_1155: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
    mul_711: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, unsqueeze_1155);  unsqueeze_1155 = None
    mul_712: "f32[224]" = torch.ops.aten.mul.Tensor(sum_128, rsqrt_54);  sum_128 = rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_711, mul_22, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_711 = mul_22 = primals_140 = None
    getitem_264: "f32[4, 224, 56, 56]" = convolution_backward_88[0]
    getitem_265: "f32[224, 224, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_713: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_264, relu_6)
    mul_714: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_264, sigmoid_1);  getitem_264 = sigmoid_1 = None
    sum_129: "f32[4, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2, 3], True);  mul_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_284: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_134: "f32[4, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_284)
    mul_715: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(alias_284, sub_134);  alias_284 = sub_134 = None
    mul_716: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_129, mul_715);  sum_129 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_716, relu_7, primals_138, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_716 = primals_138 = None
    getitem_267: "f32[4, 56, 1, 1]" = convolution_backward_89[0]
    getitem_268: "f32[224, 56, 1, 1]" = convolution_backward_89[1]
    getitem_269: "f32[224]" = convolution_backward_89[2];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_286: "f32[4, 56, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_287: "f32[4, 56, 1, 1]" = torch.ops.aten.alias.default(alias_286);  alias_286 = None
    le_69: "b8[4, 56, 1, 1]" = torch.ops.aten.le.Scalar(alias_287, 0);  alias_287 = None
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_69: "f32[4, 56, 1, 1]" = torch.ops.aten.where.self(le_69, scalar_tensor_69, getitem_267);  le_69 = scalar_tensor_69 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(where_69, mean_1, primals_136, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_69 = mean_1 = primals_136 = None
    getitem_270: "f32[4, 224, 1, 1]" = convolution_backward_90[0]
    getitem_271: "f32[56, 224, 1, 1]" = convolution_backward_90[1]
    getitem_272: "f32[56]" = convolution_backward_90[2];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_18: "f32[4, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_270, [4, 224, 56, 56]);  getitem_270 = None
    div_18: "f32[4, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_18, 3136);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_232: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_714, div_18);  mul_714 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_289: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_290: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(alias_289);  alias_289 = None
    le_70: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(alias_290, 0);  alias_290 = None
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_70: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_70, scalar_tensor_70, add_232);  le_70 = scalar_tensor_70 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_233: "f32[224]" = torch.ops.aten.add.Tensor(primals_278, 1e-05);  primals_278 = None
    rsqrt_55: "f32[224]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
    unsqueeze_1156: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_277, 0);  primals_277 = None
    unsqueeze_1157: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
    unsqueeze_1158: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
    sum_130: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_135: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1158);  convolution_8 = unsqueeze_1158 = None
    mul_717: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, sub_135);  sub_135 = None
    sum_131: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_722: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_55, primals_13);  primals_13 = None
    unsqueeze_1165: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_1166: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
    unsqueeze_1167: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
    mul_723: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, unsqueeze_1167);  where_70 = unsqueeze_1167 = None
    mul_724: "f32[224]" = torch.ops.aten.mul.Tensor(sum_131, rsqrt_55);  sum_131 = rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_723, relu_5, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_723 = primals_135 = None
    getitem_273: "f32[4, 224, 56, 56]" = convolution_backward_91[0]
    getitem_274: "f32[224, 112, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_292: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_293: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(alias_292);  alias_292 = None
    le_71: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(alias_293, 0);  alias_293 = None
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_71: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_71, scalar_tensor_71, getitem_273);  le_71 = scalar_tensor_71 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_234: "f32[224]" = torch.ops.aten.add.Tensor(primals_276, 1e-05);  primals_276 = None
    rsqrt_56: "f32[224]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    unsqueeze_1168: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_275, 0);  primals_275 = None
    unsqueeze_1169: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
    sum_132: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_136: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1170);  convolution_7 = unsqueeze_1170 = None
    mul_725: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, sub_136);  sub_136 = None
    sum_133: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_730: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_56, primals_11);  primals_11 = None
    unsqueeze_1177: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_1178: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
    unsqueeze_1179: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
    mul_731: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, unsqueeze_1179);  where_71 = unsqueeze_1179 = None
    mul_732: "f32[224]" = torch.ops.aten.mul.Tensor(sum_133, rsqrt_56);  sum_133 = rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_731, relu_4, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = relu_4 = primals_134 = None
    getitem_276: "f32[4, 224, 56, 56]" = convolution_backward_92[0]
    getitem_277: "f32[224, 224, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_235: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(where_68, getitem_276);  where_68 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_294: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    le_72: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_72: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_72, scalar_tensor_72, add_235);  le_72 = scalar_tensor_72 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_236: "f32[224]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_57: "f32[224]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    unsqueeze_1180: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_1181: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
    unsqueeze_1182: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None
    sum_134: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_137: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1182);  convolution_6 = unsqueeze_1182 = None
    mul_733: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_137);  sub_137 = None
    sum_135: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_738: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_57, primals_9);  primals_9 = None
    unsqueeze_1189: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_1190: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
    unsqueeze_1191: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
    mul_739: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, unsqueeze_1191);  unsqueeze_1191 = None
    mul_740: "f32[224]" = torch.ops.aten.mul.Tensor(sum_135, rsqrt_57);  sum_135 = rsqrt_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_739, relu, primals_133, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = primals_133 = None
    getitem_279: "f32[4, 32, 112, 112]" = convolution_backward_93[0]
    getitem_280: "f32[224, 32, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_237: "f32[224]" = torch.ops.aten.add.Tensor(primals_272, 1e-05);  primals_272 = None
    rsqrt_58: "f32[224]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    unsqueeze_1192: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_271, 0);  primals_271 = None
    unsqueeze_1193: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
    unsqueeze_1194: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None
    sum_136: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_138: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1194);  convolution_5 = unsqueeze_1194 = None
    mul_741: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_138);  sub_138 = None
    sum_137: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
    mul_746: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_58, primals_7);  primals_7 = None
    unsqueeze_1201: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_1202: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
    unsqueeze_1203: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
    mul_747: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, unsqueeze_1203);  where_72 = unsqueeze_1203 = None
    mul_748: "f32[224]" = torch.ops.aten.mul.Tensor(sum_137, rsqrt_58);  sum_137 = rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_747, mul_9, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_9 = primals_132 = None
    getitem_282: "f32[4, 224, 56, 56]" = convolution_backward_94[0]
    getitem_283: "f32[224, 224, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_749: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_282, relu_2)
    mul_750: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_282, sigmoid);  getitem_282 = sigmoid = None
    sum_138: "f32[4, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_295: "f32[4, 224, 1, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_139: "f32[4, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_295)
    mul_751: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(alias_295, sub_139);  alias_295 = sub_139 = None
    mul_752: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_138, mul_751);  sum_138 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_752, relu_3, primals_130, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_752 = primals_130 = None
    getitem_285: "f32[4, 8, 1, 1]" = convolution_backward_95[0]
    getitem_286: "f32[224, 8, 1, 1]" = convolution_backward_95[1]
    getitem_287: "f32[224]" = convolution_backward_95[2];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_297: "f32[4, 8, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_298: "f32[4, 8, 1, 1]" = torch.ops.aten.alias.default(alias_297);  alias_297 = None
    le_73: "b8[4, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_298, 0);  alias_298 = None
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_73: "f32[4, 8, 1, 1]" = torch.ops.aten.where.self(le_73, scalar_tensor_73, getitem_285);  le_73 = scalar_tensor_73 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(where_73, mean, primals_128, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_73 = mean = primals_128 = None
    getitem_288: "f32[4, 224, 1, 1]" = convolution_backward_96[0]
    getitem_289: "f32[8, 224, 1, 1]" = convolution_backward_96[1]
    getitem_290: "f32[8]" = convolution_backward_96[2];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_19: "f32[4, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_288, [4, 224, 56, 56]);  getitem_288 = None
    div_19: "f32[4, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_19, 3136);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_238: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_750, div_19);  mul_750 = div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_300: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_301: "f32[4, 224, 56, 56]" = torch.ops.aten.alias.default(alias_300);  alias_300 = None
    le_74: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(alias_301, 0);  alias_301 = None
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_74: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_74, scalar_tensor_74, add_238);  le_74 = scalar_tensor_74 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_239: "f32[224]" = torch.ops.aten.add.Tensor(primals_270, 1e-05);  primals_270 = None
    rsqrt_59: "f32[224]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    unsqueeze_1204: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_269, 0);  primals_269 = None
    unsqueeze_1205: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
    unsqueeze_1206: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None
    sum_139: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_140: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1206);  convolution_2 = unsqueeze_1206 = None
    mul_753: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, sub_140);  sub_140 = None
    sum_140: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
    mul_758: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_59, primals_5);  primals_5 = None
    unsqueeze_1213: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_1214: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
    unsqueeze_1215: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
    mul_759: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, unsqueeze_1215);  where_74 = unsqueeze_1215 = None
    mul_760: "f32[224]" = torch.ops.aten.mul.Tensor(sum_140, rsqrt_59);  sum_140 = rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_759, relu_1, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_759 = primals_127 = None
    getitem_291: "f32[4, 224, 112, 112]" = convolution_backward_97[0]
    getitem_292: "f32[224, 112, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_303: "f32[4, 224, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_304: "f32[4, 224, 112, 112]" = torch.ops.aten.alias.default(alias_303);  alias_303 = None
    le_75: "b8[4, 224, 112, 112]" = torch.ops.aten.le.Scalar(alias_304, 0);  alias_304 = None
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_75: "f32[4, 224, 112, 112]" = torch.ops.aten.where.self(le_75, scalar_tensor_75, getitem_291);  le_75 = scalar_tensor_75 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_240: "f32[224]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_60: "f32[224]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    unsqueeze_1216: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_1217: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
    unsqueeze_1218: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None
    sum_141: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_141: "f32[4, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1218);  convolution_1 = unsqueeze_1218 = None
    mul_761: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(where_75, sub_141);  sub_141 = None
    sum_142: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
    mul_766: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_60, primals_3);  primals_3 = None
    unsqueeze_1225: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_1226: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
    unsqueeze_1227: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
    mul_767: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(where_75, unsqueeze_1227);  where_75 = unsqueeze_1227 = None
    mul_768: "f32[224]" = torch.ops.aten.mul.Tensor(sum_142, rsqrt_60);  sum_142 = rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_767, relu, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_767 = primals_126 = None
    getitem_294: "f32[4, 32, 112, 112]" = convolution_backward_98[0]
    getitem_295: "f32[224, 32, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_241: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_279, getitem_294);  getitem_279 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_306: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_307: "f32[4, 32, 112, 112]" = torch.ops.aten.alias.default(alias_306);  alias_306 = None
    le_76: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_307, 0);  alias_307 = None
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_76: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_76, scalar_tensor_76, add_241);  le_76 = scalar_tensor_76 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_242: "f32[32]" = torch.ops.aten.add.Tensor(primals_266, 1e-05);  primals_266 = None
    rsqrt_61: "f32[32]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    unsqueeze_1228: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_265, 0);  primals_265 = None
    unsqueeze_1229: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
    unsqueeze_1230: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None
    sum_143: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_142: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1230);  convolution = unsqueeze_1230 = None
    mul_769: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_76, sub_142);  sub_142 = None
    sum_144: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_774: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_61, primals_1);  primals_1 = None
    unsqueeze_1237: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_1238: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
    unsqueeze_1239: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
    mul_775: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_76, unsqueeze_1239);  where_76 = unsqueeze_1239 = None
    mul_776: "f32[32]" = torch.ops.aten.mul.Tensor(sum_144, rsqrt_61);  sum_144 = rsqrt_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_775, primals_389, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_775 = primals_389 = primals_125 = None
    getitem_298: "f32[32, 3, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
    return pytree.tree_unflatten([addmm, mul_776, sum_143, mul_768, sum_141, mul_760, sum_139, mul_748, sum_136, mul_740, sum_134, mul_732, sum_132, mul_724, sum_130, mul_712, sum_127, mul_704, sum_125, mul_696, sum_123, mul_684, sum_120, mul_676, sum_118, mul_668, sum_116, mul_660, sum_114, mul_648, sum_111, mul_640, sum_109, mul_632, sum_107, mul_620, sum_104, mul_612, sum_102, mul_604, sum_100, mul_592, sum_97, mul_584, sum_95, mul_576, sum_93, mul_564, sum_90, mul_556, sum_88, mul_548, sum_86, mul_536, sum_83, mul_528, sum_81, mul_520, sum_79, mul_512, sum_77, mul_500, sum_74, mul_492, sum_72, mul_484, sum_70, mul_472, sum_67, mul_464, sum_65, mul_456, sum_63, mul_444, sum_60, mul_436, sum_58, mul_428, sum_56, mul_416, sum_53, mul_408, sum_51, mul_400, sum_49, mul_388, sum_46, mul_380, sum_44, mul_372, sum_42, mul_360, sum_39, mul_352, sum_37, mul_344, sum_35, mul_332, sum_32, mul_324, sum_30, mul_316, sum_28, mul_304, sum_25, mul_296, sum_23, mul_288, sum_21, mul_276, sum_18, mul_268, sum_16, mul_260, sum_14, mul_248, sum_11, mul_240, sum_9, mul_232, sum_7, mul_220, sum_4, mul_212, sum_2, getitem_298, getitem_295, getitem_292, getitem_289, getitem_290, getitem_286, getitem_287, getitem_283, getitem_280, getitem_277, getitem_274, getitem_271, getitem_272, getitem_268, getitem_269, getitem_265, getitem_262, getitem_259, getitem_256, getitem_257, getitem_253, getitem_254, getitem_250, getitem_247, getitem_244, getitem_241, getitem_238, getitem_239, getitem_235, getitem_236, getitem_232, getitem_229, getitem_226, getitem_223, getitem_224, getitem_220, getitem_221, getitem_217, getitem_214, getitem_211, getitem_208, getitem_209, getitem_205, getitem_206, getitem_202, getitem_199, getitem_196, getitem_193, getitem_194, getitem_190, getitem_191, getitem_187, getitem_184, getitem_181, getitem_178, getitem_179, getitem_175, getitem_176, getitem_172, getitem_169, getitem_166, getitem_163, getitem_160, getitem_161, getitem_157, getitem_158, getitem_154, getitem_151, getitem_148, getitem_145, getitem_146, getitem_142, getitem_143, getitem_139, getitem_136, getitem_133, getitem_130, getitem_131, getitem_127, getitem_128, getitem_124, getitem_121, getitem_118, getitem_115, getitem_116, getitem_112, getitem_113, getitem_109, getitem_106, getitem_103, getitem_100, getitem_101, getitem_97, getitem_98, getitem_94, getitem_91, getitem_88, getitem_85, getitem_86, getitem_82, getitem_83, getitem_79, getitem_76, getitem_73, getitem_70, getitem_71, getitem_67, getitem_68, getitem_64, getitem_61, getitem_58, getitem_55, getitem_56, getitem_52, getitem_53, getitem_49, getitem_46, getitem_43, getitem_40, getitem_41, getitem_37, getitem_38, getitem_34, getitem_31, getitem_28, getitem_25, getitem_26, getitem_22, getitem_23, getitem_19, getitem_16, getitem_13, getitem_10, getitem_11, getitem_7, getitem_8, getitem_4, getitem_1, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    