from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[29056, 1024]"; primals_2: "f32[2, 1024]"; primals_3: "f32[512, 1024]"; primals_4: "f32[1024]"; primals_5: "f32[1024]"; primals_6: "f32[1024, 1024]"; primals_7: "f32[1024]"; primals_8: "f32[1024, 1024]"; primals_9: "f32[1024]"; primals_10: "f32[1024, 1024]"; primals_11: "f32[1024]"; primals_12: "f32[1024, 1024]"; primals_13: "f32[1024]"; primals_14: "f32[1024]"; primals_15: "f32[1024]"; primals_16: "f32[4096, 1024]"; primals_17: "f32[4096]"; primals_18: "f32[1024, 4096]"; primals_19: "f32[1024]"; primals_20: "f32[1024]"; primals_21: "f32[1024]"; primals_22: "f32[1024, 1024]"; primals_23: "f32[1024]"; primals_24: "f32[1024, 1024]"; primals_25: "f32[1024]"; primals_26: "f32[1024, 1024]"; primals_27: "f32[1024]"; primals_28: "f32[1024, 1024]"; primals_29: "f32[1024]"; primals_30: "f32[1024]"; primals_31: "f32[1024]"; primals_32: "f32[4096, 1024]"; primals_33: "f32[4096]"; primals_34: "f32[1024, 4096]"; primals_35: "f32[1024]"; primals_36: "f32[1024]"; primals_37: "f32[1024]"; primals_38: "f32[1024, 1024]"; primals_39: "f32[1024]"; primals_40: "f32[1024, 1024]"; primals_41: "f32[1024]"; primals_42: "f32[1024, 1024]"; primals_43: "f32[1024]"; primals_44: "f32[1024, 1024]"; primals_45: "f32[1024]"; primals_46: "f32[1024]"; primals_47: "f32[1024]"; primals_48: "f32[4096, 1024]"; primals_49: "f32[4096]"; primals_50: "f32[1024, 4096]"; primals_51: "f32[1024]"; primals_52: "f32[1024]"; primals_53: "f32[1024]"; primals_54: "f32[1024, 1024]"; primals_55: "f32[1024]"; primals_56: "f32[1024, 1024]"; primals_57: "f32[1024]"; primals_58: "f32[1024, 1024]"; primals_59: "f32[1024]"; primals_60: "f32[1024, 1024]"; primals_61: "f32[1024]"; primals_62: "f32[1024]"; primals_63: "f32[1024]"; primals_64: "f32[4096, 1024]"; primals_65: "f32[4096]"; primals_66: "f32[1024, 4096]"; primals_67: "f32[1024]"; primals_68: "f32[1024]"; primals_69: "f32[1024]"; primals_70: "f32[1024, 1024]"; primals_71: "f32[1024]"; primals_72: "f32[1024, 1024]"; primals_73: "f32[1024]"; primals_74: "f32[1024, 1024]"; primals_75: "f32[1024]"; primals_76: "f32[1024, 1024]"; primals_77: "f32[1024]"; primals_78: "f32[1024]"; primals_79: "f32[1024]"; primals_80: "f32[4096, 1024]"; primals_81: "f32[4096]"; primals_82: "f32[1024, 4096]"; primals_83: "f32[1024]"; primals_84: "f32[1024]"; primals_85: "f32[1024]"; primals_86: "f32[1024, 1024]"; primals_87: "f32[1024]"; primals_88: "f32[1024, 1024]"; primals_89: "f32[1024]"; primals_90: "f32[1024, 1024]"; primals_91: "f32[1024]"; primals_92: "f32[1024, 1024]"; primals_93: "f32[1024]"; primals_94: "f32[1024]"; primals_95: "f32[1024]"; primals_96: "f32[4096, 1024]"; primals_97: "f32[4096]"; primals_98: "f32[1024, 4096]"; primals_99: "f32[1024]"; primals_100: "f32[1024]"; primals_101: "f32[1024]"; primals_102: "f32[1024, 1024]"; primals_103: "f32[1024]"; primals_104: "f32[1024, 1024]"; primals_105: "f32[1024]"; primals_106: "f32[1024, 1024]"; primals_107: "f32[1024]"; primals_108: "f32[1024, 1024]"; primals_109: "f32[1024]"; primals_110: "f32[1024]"; primals_111: "f32[1024]"; primals_112: "f32[4096, 1024]"; primals_113: "f32[4096]"; primals_114: "f32[1024, 4096]"; primals_115: "f32[1024]"; primals_116: "f32[1024]"; primals_117: "f32[1024]"; primals_118: "f32[1024, 1024]"; primals_119: "f32[1024]"; primals_120: "f32[1024, 1024]"; primals_121: "f32[1024]"; primals_122: "f32[1024, 1024]"; primals_123: "f32[1024]"; primals_124: "f32[1024, 1024]"; primals_125: "f32[1024]"; primals_126: "f32[1024]"; primals_127: "f32[1024]"; primals_128: "f32[4096, 1024]"; primals_129: "f32[4096]"; primals_130: "f32[1024, 4096]"; primals_131: "f32[1024]"; primals_132: "f32[1024]"; primals_133: "f32[1024]"; primals_134: "f32[1024, 1024]"; primals_135: "f32[1024]"; primals_136: "f32[1024, 1024]"; primals_137: "f32[1024]"; primals_138: "f32[1024, 1024]"; primals_139: "f32[1024]"; primals_140: "f32[1024, 1024]"; primals_141: "f32[1024]"; primals_142: "f32[1024]"; primals_143: "f32[1024]"; primals_144: "f32[4096, 1024]"; primals_145: "f32[4096]"; primals_146: "f32[1024, 4096]"; primals_147: "f32[1024]"; primals_148: "f32[1024]"; primals_149: "f32[1024]"; primals_150: "f32[1024, 1024]"; primals_151: "f32[1024]"; primals_152: "f32[1024, 1024]"; primals_153: "f32[1024]"; primals_154: "f32[1024, 1024]"; primals_155: "f32[1024]"; primals_156: "f32[1024, 1024]"; primals_157: "f32[1024]"; primals_158: "f32[1024]"; primals_159: "f32[1024]"; primals_160: "f32[4096, 1024]"; primals_161: "f32[4096]"; primals_162: "f32[1024, 4096]"; primals_163: "f32[1024]"; primals_164: "f32[1024]"; primals_165: "f32[1024]"; primals_166: "f32[1024, 1024]"; primals_167: "f32[1024]"; primals_168: "f32[1024, 1024]"; primals_169: "f32[1024]"; primals_170: "f32[1024, 1024]"; primals_171: "f32[1024]"; primals_172: "f32[1024, 1024]"; primals_173: "f32[1024]"; primals_174: "f32[1024]"; primals_175: "f32[1024]"; primals_176: "f32[4096, 1024]"; primals_177: "f32[4096]"; primals_178: "f32[1024, 4096]"; primals_179: "f32[1024]"; primals_180: "f32[1024]"; primals_181: "f32[1024]"; primals_182: "f32[1024, 1024]"; primals_183: "f32[1024]"; primals_184: "f32[1024, 1024]"; primals_185: "f32[1024]"; primals_186: "f32[1024, 1024]"; primals_187: "f32[1024]"; primals_188: "f32[1024, 1024]"; primals_189: "f32[1024]"; primals_190: "f32[1024]"; primals_191: "f32[1024]"; primals_192: "f32[4096, 1024]"; primals_193: "f32[4096]"; primals_194: "f32[1024, 4096]"; primals_195: "f32[1024]"; primals_196: "f32[1024]"; primals_197: "f32[1024]"; primals_198: "f32[1024, 1024]"; primals_199: "f32[1024]"; primals_200: "f32[1024, 1024]"; primals_201: "f32[1024]"; primals_202: "f32[1024, 1024]"; primals_203: "f32[1024]"; primals_204: "f32[1024, 1024]"; primals_205: "f32[1024]"; primals_206: "f32[1024]"; primals_207: "f32[1024]"; primals_208: "f32[4096, 1024]"; primals_209: "f32[4096]"; primals_210: "f32[1024, 4096]"; primals_211: "f32[1024]"; primals_212: "f32[1024]"; primals_213: "f32[1024]"; primals_214: "f32[1024, 1024]"; primals_215: "f32[1024]"; primals_216: "f32[1024, 1024]"; primals_217: "f32[1024]"; primals_218: "f32[1024, 1024]"; primals_219: "f32[1024]"; primals_220: "f32[1024, 1024]"; primals_221: "f32[1024]"; primals_222: "f32[1024]"; primals_223: "f32[1024]"; primals_224: "f32[4096, 1024]"; primals_225: "f32[4096]"; primals_226: "f32[1024, 4096]"; primals_227: "f32[1024]"; primals_228: "f32[1024]"; primals_229: "f32[1024]"; primals_230: "f32[1024, 1024]"; primals_231: "f32[1024]"; primals_232: "f32[1024, 1024]"; primals_233: "f32[1024]"; primals_234: "f32[1024, 1024]"; primals_235: "f32[1024]"; primals_236: "f32[1024, 1024]"; primals_237: "f32[1024]"; primals_238: "f32[1024]"; primals_239: "f32[1024]"; primals_240: "f32[4096, 1024]"; primals_241: "f32[4096]"; primals_242: "f32[1024, 4096]"; primals_243: "f32[1024]"; primals_244: "f32[1024]"; primals_245: "f32[1024]"; primals_246: "f32[1024, 1024]"; primals_247: "f32[1024]"; primals_248: "f32[1024, 1024]"; primals_249: "f32[1024]"; primals_250: "f32[1024, 1024]"; primals_251: "f32[1024]"; primals_252: "f32[1024, 1024]"; primals_253: "f32[1024]"; primals_254: "f32[1024]"; primals_255: "f32[1024]"; primals_256: "f32[4096, 1024]"; primals_257: "f32[4096]"; primals_258: "f32[1024, 4096]"; primals_259: "f32[1024]"; primals_260: "f32[1024]"; primals_261: "f32[1024]"; primals_262: "f32[1024, 1024]"; primals_263: "f32[1024]"; primals_264: "f32[1024, 1024]"; primals_265: "f32[1024]"; primals_266: "f32[1024, 1024]"; primals_267: "f32[1024]"; primals_268: "f32[1024, 1024]"; primals_269: "f32[1024]"; primals_270: "f32[1024]"; primals_271: "f32[1024]"; primals_272: "f32[4096, 1024]"; primals_273: "f32[4096]"; primals_274: "f32[1024, 4096]"; primals_275: "f32[1024]"; primals_276: "f32[1024]"; primals_277: "f32[1024]"; primals_278: "f32[1024, 1024]"; primals_279: "f32[1024]"; primals_280: "f32[1024, 1024]"; primals_281: "f32[1024]"; primals_282: "f32[1024, 1024]"; primals_283: "f32[1024]"; primals_284: "f32[1024, 1024]"; primals_285: "f32[1024]"; primals_286: "f32[1024]"; primals_287: "f32[1024]"; primals_288: "f32[4096, 1024]"; primals_289: "f32[4096]"; primals_290: "f32[1024, 4096]"; primals_291: "f32[1024]"; primals_292: "f32[1024]"; primals_293: "f32[1024]"; primals_294: "f32[1024, 1024]"; primals_295: "f32[1024]"; primals_296: "f32[1024, 1024]"; primals_297: "f32[1024]"; primals_298: "f32[1024, 1024]"; primals_299: "f32[1024]"; primals_300: "f32[1024, 1024]"; primals_301: "f32[1024]"; primals_302: "f32[1024]"; primals_303: "f32[1024]"; primals_304: "f32[4096, 1024]"; primals_305: "f32[4096]"; primals_306: "f32[1024, 4096]"; primals_307: "f32[1024]"; primals_308: "f32[1024]"; primals_309: "f32[1024]"; primals_310: "f32[1024, 1024]"; primals_311: "f32[1024]"; primals_312: "f32[1024, 1024]"; primals_313: "f32[1024]"; primals_314: "f32[1024, 1024]"; primals_315: "f32[1024]"; primals_316: "f32[1024, 1024]"; primals_317: "f32[1024]"; primals_318: "f32[1024]"; primals_319: "f32[1024]"; primals_320: "f32[4096, 1024]"; primals_321: "f32[4096]"; primals_322: "f32[1024, 4096]"; primals_323: "f32[1024]"; primals_324: "f32[1024]"; primals_325: "f32[1024]"; primals_326: "f32[1024, 1024]"; primals_327: "f32[1024]"; primals_328: "f32[1024, 1024]"; primals_329: "f32[1024]"; primals_330: "f32[1024, 1024]"; primals_331: "f32[1024]"; primals_332: "f32[1024, 1024]"; primals_333: "f32[1024]"; primals_334: "f32[1024]"; primals_335: "f32[1024]"; primals_336: "f32[4096, 1024]"; primals_337: "f32[4096]"; primals_338: "f32[1024, 4096]"; primals_339: "f32[1024]"; primals_340: "f32[1024]"; primals_341: "f32[1024]"; primals_342: "f32[1024, 1024]"; primals_343: "f32[1024]"; primals_344: "f32[1024, 1024]"; primals_345: "f32[1024]"; primals_346: "f32[1024, 1024]"; primals_347: "f32[1024]"; primals_348: "f32[1024, 1024]"; primals_349: "f32[1024]"; primals_350: "f32[1024]"; primals_351: "f32[1024]"; primals_352: "f32[4096, 1024]"; primals_353: "f32[4096]"; primals_354: "f32[1024, 4096]"; primals_355: "f32[1024]"; primals_356: "f32[1024]"; primals_357: "f32[1024]"; primals_358: "f32[1024, 1024]"; primals_359: "f32[1024]"; primals_360: "f32[1024, 1024]"; primals_361: "f32[1024]"; primals_362: "f32[1024, 1024]"; primals_363: "f32[1024]"; primals_364: "f32[1024, 1024]"; primals_365: "f32[1024]"; primals_366: "f32[1024]"; primals_367: "f32[1024]"; primals_368: "f32[4096, 1024]"; primals_369: "f32[4096]"; primals_370: "f32[1024, 4096]"; primals_371: "f32[1024]"; primals_372: "f32[1024]"; primals_373: "f32[1024]"; primals_374: "f32[1024, 1024]"; primals_375: "f32[1024]"; primals_376: "f32[1024, 1024]"; primals_377: "f32[1024]"; primals_378: "f32[1024, 1024]"; primals_379: "f32[1024]"; primals_380: "f32[1024, 1024]"; primals_381: "f32[1024]"; primals_382: "f32[1024]"; primals_383: "f32[1024]"; primals_384: "f32[4096, 1024]"; primals_385: "f32[4096]"; primals_386: "f32[1024, 4096]"; primals_387: "f32[1024]"; primals_388: "f32[1024]"; primals_389: "f32[1024]"; primals_390: "f32[1024, 1024]"; primals_391: "f32[1024]"; primals_392: "f32[1024]"; primals_393: "f32[1024]"; primals_394: "f32[29056, 1024]"; primals_395: "f32[29056]"; primals_396: "i64[1, 512]"; primals_397: "i64[1, 512]"; primals_398: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 29056]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:950, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:952, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_1: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_1: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_2: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_2);  slice_2 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:173, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_3: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_396, 0, 0, 9223372036854775807);  primals_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_1, primals_398, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_2, full_1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:182, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:184, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(primals_3, slice_3);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_1, 0.1, True);  add_1 = None
    getitem: "f32[1, 512, 1024]" = native_dropout[0]
    getitem_1: "b8[1, 512, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean[0]
    getitem_3: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(getitem, getitem_3)
    mul_1: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
    add_3: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 1024]" = torch.ops.aten.view.default(add_3, [512, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm, [1, 512, 1024]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_2: "f32[512, 1024]" = torch.ops.aten.view.default(add_3, [512, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 512, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 16, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_5: "f32[512, 1024]" = torch.ops.aten.view.default(add_3, [512, 1024]);  add_3 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_11, view_5, permute_3);  primals_11 = None
    view_6: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 512, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_6, [1, 512, 16, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 16, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_6: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
    expand: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 16, 512, 64]);  permute_5 = None
    view_9: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand, [16, 512, 64]);  expand = None
    expand_1: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_6, [1, 16, 64, 512]);  permute_6 = None
    view_10: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_1, [16, 64, 512]);  expand_1 = None
    bmm: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 16, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_4: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
    sub_2: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
    exp: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 16, 512, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 16, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_2: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_4, [1, 16, 512, 512]);  getitem_4 = None
    view_12: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_2, [16, 512, 512]);  expand_2 = None
    expand_3: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_4, [1, 16, 512, 64]);  permute_4 = None
    view_13: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_3, [16, 512, 64]);  expand_3 = None
    bmm_1: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone, [1, 512, 1024]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 1024]" = torch.ops.aten.view.default(view_15, [512, 1024]);  view_15 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
    view_17: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 512, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 1024]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 1024]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_5: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(getitem, getitem_6);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_9)
    mul_3: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_14);  mul_3 = None
    add_7: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_15);  mul_4 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 1024]" = torch.ops.aten.view.default(add_7, [512, 1024]);  add_7 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
    view_19: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 512, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 4096]" = torch.ops.aten.view.default(mul_7, [512, 4096]);  mul_7 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 1024]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 1024]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_9: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_5, getitem_10);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_13)
    mul_8: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_8, primals_20);  mul_8 = None
    add_11: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_9, primals_21);  mul_9 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 1024]" = torch.ops.aten.view.default(add_11, [512, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
    view_23: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 512, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_24: "f32[512, 1024]" = torch.ops.aten.view.default(add_11, [512, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_25, view_24, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 512, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 16, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_27: "f32[512, 1024]" = torch.ops.aten.view.default(add_11, [512, 1024]);  add_11 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_27, view_27, permute_14);  primals_27 = None
    view_28: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_8, [1, 512, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 16, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 16, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
    expand_4: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 16, 512, 64]);  permute_16 = None
    view_31: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_4, [16, 512, 64]);  expand_4 = None
    expand_5: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 16, 64, 512]);  permute_17 = None
    view_32: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_5, [16, 64, 512]);  expand_5 = None
    bmm_2: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 16, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_12: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_12, [-1], True)
    sub_5: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_12, amax_1);  add_12 = amax_1 = None
    exp_1: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_14: "f32[1, 16, 512, 512]" = native_dropout_4[0]
    getitem_15: "b8[1, 16, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_6: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_14, [1, 16, 512, 512]);  getitem_14 = None
    view_34: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_6, [16, 512, 512]);  expand_6 = None
    expand_7: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 16, 512, 64]);  permute_15 = None
    view_35: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_7, [16, 512, 64]);  expand_7 = None
    bmm_3: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 16, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_1: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_1, [1, 512, 1024]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 1024]" = torch.ops.aten.view.default(view_37, [512, 1024]);  view_37 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
    view_39: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 512, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 1024]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 1024]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_13: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_9, getitem_16);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_19)
    mul_10: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, primals_30);  mul_10 = None
    add_15: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_11, primals_31);  mul_11 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 1024]" = torch.ops.aten.view.default(add_15, [512, 1024]);  add_15 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
    view_41: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 512, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 4096]" = torch.ops.aten.view.default(mul_14, [512, 4096]);  mul_14 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 1024]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 1024]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_17: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_13, getitem_20);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_23)
    mul_15: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, primals_36);  mul_15 = None
    add_19: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_16, primals_37);  mul_16 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
    view_45: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_12, [1, 512, 1024]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_46: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_41, view_46, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_13, [1, 512, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 16, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_49: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024]);  add_19 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_43, view_49, permute_25);  primals_43 = None
    view_50: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_14, [1, 512, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_50, [1, 512, 16, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 16, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_28: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
    expand_8: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 16, 512, 64]);  permute_27 = None
    view_53: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_8, [16, 512, 64]);  expand_8 = None
    expand_9: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 16, 64, 512]);  permute_28 = None
    view_54: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_9, [16, 64, 512]);  expand_9 = None
    bmm_4: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 16, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_20: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_20, [-1], True)
    sub_8: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_20, amax_2);  add_20 = amax_2 = None
    exp_2: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_24: "f32[1, 16, 512, 512]" = native_dropout_7[0]
    getitem_25: "b8[1, 16, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_10: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_24, [1, 16, 512, 512]);  getitem_24 = None
    view_56: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_10, [16, 512, 512]);  expand_10 = None
    expand_11: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 16, 512, 64]);  permute_26 = None
    view_57: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_11, [16, 512, 64]);  expand_11 = None
    bmm_5: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 16, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_2: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_2, [1, 512, 1024]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 1024]" = torch.ops.aten.view.default(view_59, [512, 1024]);  view_59 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
    view_61: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_15, [1, 512, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 1024]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 1024]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_21: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_17, getitem_26);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_29)
    mul_17: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, primals_46);  mul_17 = None
    add_23: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_18, primals_47);  mul_18 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 1024]" = torch.ops.aten.view.default(add_23, [512, 1024]);  add_23 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
    view_63: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 512, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 4096]" = torch.ops.aten.view.default(mul_21, [512, 4096]);  mul_21 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 1024]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 1024]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_25: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_21, getitem_30);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_33)
    mul_22: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, primals_52);  mul_22 = None
    add_27: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_23, primals_53);  mul_23 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 1024]" = torch.ops.aten.view.default(add_27, [512, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
    view_67: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_18, [1, 512, 1024]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_68: "f32[512, 1024]" = torch.ops.aten.view.default(add_27, [512, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_19, [1, 512, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 16, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_71: "f32[512, 1024]" = torch.ops.aten.view.default(add_27, [512, 1024]);  add_27 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_59, view_71, permute_36);  primals_59 = None
    view_72: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_20, [1, 512, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_72, [1, 512, 16, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 16, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_39: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
    expand_12: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 16, 512, 64]);  permute_38 = None
    view_75: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_12, [16, 512, 64]);  expand_12 = None
    expand_13: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_39, [1, 16, 64, 512]);  permute_39 = None
    view_76: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_13, [16, 64, 512]);  expand_13 = None
    bmm_6: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 16, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_28: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
    sub_11: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_28, amax_3);  add_28 = amax_3 = None
    exp_3: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_34: "f32[1, 16, 512, 512]" = native_dropout_10[0]
    getitem_35: "b8[1, 16, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_14: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_34, [1, 16, 512, 512]);  getitem_34 = None
    view_78: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_14, [16, 512, 512]);  expand_14 = None
    expand_15: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_37, [1, 16, 512, 64]);  permute_37 = None
    view_79: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_15, [16, 512, 64]);  expand_15 = None
    bmm_7: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 16, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_3: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_3, [1, 512, 1024]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 1024]" = torch.ops.aten.view.default(view_81, [512, 1024]);  view_81 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
    view_83: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_21, [1, 512, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 1024]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 1024]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_29: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_25, getitem_36);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_39)
    mul_24: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, primals_62);  mul_24 = None
    add_31: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_25, primals_63);  mul_25 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 1024]" = torch.ops.aten.view.default(add_31, [512, 1024]);  add_31 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
    view_85: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 512, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 4096]" = torch.ops.aten.view.default(mul_28, [512, 4096]);  mul_28 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 1024]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 1024]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_33: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_29, getitem_40);  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_43)
    mul_29: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, primals_68);  mul_29 = None
    add_35: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_30, primals_69);  mul_30 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 1024]" = torch.ops.aten.view.default(add_35, [512, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
    view_89: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_24, [1, 512, 1024]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_90: "f32[512, 1024]" = torch.ops.aten.view.default(add_35, [512, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_73, view_90, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_25, [1, 512, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 16, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_93: "f32[512, 1024]" = torch.ops.aten.view.default(add_35, [512, 1024]);  add_35 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_75, view_93, permute_47);  primals_75 = None
    view_94: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_26, [1, 512, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_94, [1, 512, 16, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 16, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_50: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
    expand_16: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 16, 512, 64]);  permute_49 = None
    view_97: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_16, [16, 512, 64]);  expand_16 = None
    expand_17: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_50, [1, 16, 64, 512]);  permute_50 = None
    view_98: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_17, [16, 64, 512]);  expand_17 = None
    bmm_8: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 16, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_36: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_36, [-1], True)
    sub_14: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_36, amax_4);  add_36 = amax_4 = None
    exp_4: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_44: "f32[1, 16, 512, 512]" = native_dropout_13[0]
    getitem_45: "b8[1, 16, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_18: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_44, [1, 16, 512, 512]);  getitem_44 = None
    view_100: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_18, [16, 512, 512]);  expand_18 = None
    expand_19: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_48, [1, 16, 512, 64]);  permute_48 = None
    view_101: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_19, [16, 512, 64]);  expand_19 = None
    bmm_9: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 16, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_4: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_4, [1, 512, 1024]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 1024]" = torch.ops.aten.view.default(view_103, [512, 1024]);  view_103 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
    view_105: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_27, [1, 512, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 1024]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 1024]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_37: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_33, getitem_46);  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_49)
    mul_31: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, primals_78);  mul_31 = None
    add_39: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_32, primals_79);  mul_32 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 1024]" = torch.ops.aten.view.default(add_39, [512, 1024]);  add_39 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
    view_107: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 512, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 4096]" = torch.ops.aten.view.default(mul_35, [512, 4096]);  mul_35 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 1024]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 1024]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_41: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_37, getitem_50);  getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_53)
    mul_36: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, primals_84);  mul_36 = None
    add_43: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_37, primals_85);  mul_37 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(add_43, [512, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
    view_111: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_30, [1, 512, 1024]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_112: "f32[512, 1024]" = torch.ops.aten.view.default(add_43, [512, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_31, [1, 512, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 16, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_115: "f32[512, 1024]" = torch.ops.aten.view.default(add_43, [512, 1024]);  add_43 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_91, view_115, permute_58);  primals_91 = None
    view_116: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_32, [1, 512, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 16, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 16, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_61: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
    expand_20: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 16, 512, 64]);  permute_60 = None
    view_119: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_20, [16, 512, 64]);  expand_20 = None
    expand_21: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_61, [1, 16, 64, 512]);  permute_61 = None
    view_120: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_21, [16, 64, 512]);  expand_21 = None
    bmm_10: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 16, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_44: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_44, [-1], True)
    sub_17: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_44, amax_5);  add_44 = amax_5 = None
    exp_5: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_54: "f32[1, 16, 512, 512]" = native_dropout_16[0]
    getitem_55: "b8[1, 16, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_22: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_54, [1, 16, 512, 512]);  getitem_54 = None
    view_122: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_22, [16, 512, 512]);  expand_22 = None
    expand_23: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_59, [1, 16, 512, 64]);  permute_59 = None
    view_123: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_23, [16, 512, 64]);  expand_23 = None
    bmm_11: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
    view_124: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 16, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_5: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_5, [1, 512, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 1024]" = torch.ops.aten.view.default(view_125, [512, 1024]);  view_125 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
    view_127: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_33, [1, 512, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 1024]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 1024]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_45: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_41, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_59)
    mul_38: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_38, primals_94);  mul_38 = None
    add_47: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_39, primals_95);  mul_39 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 1024]" = torch.ops.aten.view.default(add_47, [512, 1024]);  add_47 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
    view_129: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 512, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 4096]" = torch.ops.aten.view.default(mul_42, [512, 4096]);  mul_42 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 1024]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 1024]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_49: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_45, getitem_60);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_63)
    mul_43: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_43, primals_100);  mul_43 = None
    add_51: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_44, primals_101);  mul_44 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 1024]" = torch.ops.aten.view.default(add_51, [512, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_36: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
    view_133: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_36, [1, 512, 1024]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_134: "f32[512, 1024]" = torch.ops.aten.view.default(add_51, [512, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_105, view_134, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_37, [1, 512, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 16, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_137: "f32[512, 1024]" = torch.ops.aten.view.default(add_51, [512, 1024]);  add_51 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_107, view_137, permute_69);  primals_107 = None
    view_138: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_38, [1, 512, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_138, [1, 512, 16, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 16, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_72: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
    expand_24: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 16, 512, 64]);  permute_71 = None
    view_141: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_24, [16, 512, 64]);  expand_24 = None
    expand_25: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_72, [1, 16, 64, 512]);  permute_72 = None
    view_142: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_25, [16, 64, 512]);  expand_25 = None
    bmm_12: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 16, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_52: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
    sub_20: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_52, amax_6);  add_52 = amax_6 = None
    exp_6: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_64: "f32[1, 16, 512, 512]" = native_dropout_19[0]
    getitem_65: "b8[1, 16, 512, 512]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_26: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_64, [1, 16, 512, 512]);  getitem_64 = None
    view_144: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_26, [16, 512, 512]);  expand_26 = None
    expand_27: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_70, [1, 16, 512, 64]);  permute_70 = None
    view_145: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_27, [16, 512, 64]);  expand_27 = None
    bmm_13: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 16, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_6: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_6, [1, 512, 1024]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 1024]" = torch.ops.aten.view.default(view_147, [512, 1024]);  view_147 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
    view_149: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_39, [1, 512, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 1024]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 1024]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_53: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_49, getitem_66);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_69)
    mul_45: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, primals_110);  mul_45 = None
    add_55: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_46, primals_111);  mul_46 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 1024]" = torch.ops.aten.view.default(add_55, [512, 1024]);  add_55 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_40: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
    view_151: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 512, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 4096]" = torch.ops.aten.view.default(mul_49, [512, 4096]);  mul_49 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 1024]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 1024]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_57: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_53, getitem_70);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_73)
    mul_50: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_50, primals_116);  mul_50 = None
    add_59: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_51, primals_117);  mul_51 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 1024]" = torch.ops.aten.view.default(add_59, [512, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_42: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
    view_155: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_42, [1, 512, 1024]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_156: "f32[512, 1024]" = torch.ops.aten.view.default(add_59, [512, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_121, view_156, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_43, [1, 512, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 16, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_159: "f32[512, 1024]" = torch.ops.aten.view.default(add_59, [512, 1024]);  add_59 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_123, view_159, permute_80);  primals_123 = None
    view_160: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_44, [1, 512, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_160, [1, 512, 16, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 16, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_83: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
    expand_28: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 16, 512, 64]);  permute_82 = None
    view_163: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_28, [16, 512, 64]);  expand_28 = None
    expand_29: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_83, [1, 16, 64, 512]);  permute_83 = None
    view_164: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_29, [16, 64, 512]);  expand_29 = None
    bmm_14: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 16, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_60: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
    sub_23: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_60, amax_7);  add_60 = amax_7 = None
    exp_7: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_74: "f32[1, 16, 512, 512]" = native_dropout_22[0]
    getitem_75: "b8[1, 16, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_30: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_74, [1, 16, 512, 512]);  getitem_74 = None
    view_166: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_30, [16, 512, 512]);  expand_30 = None
    expand_31: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 16, 512, 64]);  permute_81 = None
    view_167: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_31, [16, 512, 64]);  expand_31 = None
    bmm_15: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 16, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_7: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_7, [1, 512, 1024]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 1024]" = torch.ops.aten.view.default(view_169, [512, 1024]);  view_169 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
    view_171: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_45, [1, 512, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 1024]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 1024]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_61: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_57, getitem_76);  getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_79)
    mul_52: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_52, primals_126);  mul_52 = None
    add_63: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_53, primals_127);  mul_53 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 1024]" = torch.ops.aten.view.default(add_63, [512, 1024]);  add_63 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_46: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
    view_173: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 512, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 4096]" = torch.ops.aten.view.default(mul_56, [512, 4096]);  mul_56 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 1024]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 1024]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_65: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_61, getitem_80);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_83)
    mul_57: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, primals_132);  mul_57 = None
    add_67: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_58, primals_133);  mul_58 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 1024]" = torch.ops.aten.view.default(add_67, [512, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_48: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
    view_177: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_48, [1, 512, 1024]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_178: "f32[512, 1024]" = torch.ops.aten.view.default(add_67, [512, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_137, view_178, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_49, [1, 512, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 16, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_181: "f32[512, 1024]" = torch.ops.aten.view.default(add_67, [512, 1024]);  add_67 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_139, view_181, permute_91);  primals_139 = None
    view_182: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_50, [1, 512, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_182, [1, 512, 16, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 16, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_94: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
    expand_32: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 16, 512, 64]);  permute_93 = None
    view_185: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_32, [16, 512, 64]);  expand_32 = None
    expand_33: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_94, [1, 16, 64, 512]);  permute_94 = None
    view_186: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_33, [16, 64, 512]);  expand_33 = None
    bmm_16: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 16, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_68: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
    sub_26: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_68, amax_8);  add_68 = amax_8 = None
    exp_8: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_84: "f32[1, 16, 512, 512]" = native_dropout_25[0]
    getitem_85: "b8[1, 16, 512, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_34: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_84, [1, 16, 512, 512]);  getitem_84 = None
    view_188: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_34, [16, 512, 512]);  expand_34 = None
    expand_35: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_92, [1, 16, 512, 64]);  permute_92 = None
    view_189: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_35, [16, 512, 64]);  expand_35 = None
    bmm_17: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 16, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_8: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_8, [1, 512, 1024]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 1024]" = torch.ops.aten.view.default(view_191, [512, 1024]);  view_191 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
    view_193: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_51, [1, 512, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 1024]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 1024]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_69: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_65, getitem_86);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_89)
    mul_59: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, primals_142);  mul_59 = None
    add_71: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_60, primals_143);  mul_60 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 1024]" = torch.ops.aten.view.default(add_71, [512, 1024]);  add_71 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_52: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
    view_195: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 512, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 4096]" = torch.ops.aten.view.default(mul_63, [512, 4096]);  mul_63 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 1024]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 1024]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_73: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_69, getitem_90);  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_93)
    mul_64: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_64, primals_148);  mul_64 = None
    add_75: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_65, primals_149);  mul_65 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 1024]" = torch.ops.aten.view.default(add_75, [512, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_54: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
    view_199: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_54, [1, 512, 1024]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_200: "f32[512, 1024]" = torch.ops.aten.view.default(add_75, [512, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_153, view_200, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_55, [1, 512, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 16, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_203: "f32[512, 1024]" = torch.ops.aten.view.default(add_75, [512, 1024]);  add_75 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_155, view_203, permute_102);  primals_155 = None
    view_204: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_56, [1, 512, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_204, [1, 512, 16, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 16, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_105: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_36: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 16, 512, 64]);  permute_104 = None
    view_207: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_36, [16, 512, 64]);  expand_36 = None
    expand_37: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_105, [1, 16, 64, 512]);  permute_105 = None
    view_208: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_37, [16, 64, 512]);  expand_37 = None
    bmm_18: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 16, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_76: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
    sub_29: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_76, amax_9);  add_76 = amax_9 = None
    exp_9: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_94: "f32[1, 16, 512, 512]" = native_dropout_28[0]
    getitem_95: "b8[1, 16, 512, 512]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_38: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_94, [1, 16, 512, 512]);  getitem_94 = None
    view_210: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_38, [16, 512, 512]);  expand_38 = None
    expand_39: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_103, [1, 16, 512, 64]);  permute_103 = None
    view_211: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_39, [16, 512, 64]);  expand_39 = None
    bmm_19: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 16, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_9: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_9, [1, 512, 1024]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 1024]" = torch.ops.aten.view.default(view_213, [512, 1024]);  view_213 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
    view_215: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_57, [1, 512, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 1024]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 1024]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_77: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_73, getitem_96);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_99)
    mul_66: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, primals_158);  mul_66 = None
    add_79: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_67, primals_159);  mul_67 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 1024]" = torch.ops.aten.view.default(add_79, [512, 1024]);  add_79 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_58: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
    view_217: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 512, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 4096]" = torch.ops.aten.view.default(mul_70, [512, 4096]);  mul_70 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 1024]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 1024]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_81: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_77, getitem_100);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_103)
    mul_71: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_71, primals_164);  mul_71 = None
    add_83: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_72, primals_165);  mul_72 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 1024]" = torch.ops.aten.view.default(add_83, [512, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_60: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
    view_221: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_60, [1, 512, 1024]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_222: "f32[512, 1024]" = torch.ops.aten.view.default(add_83, [512, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_169, view_222, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_61, [1, 512, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 16, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_225: "f32[512, 1024]" = torch.ops.aten.view.default(add_83, [512, 1024]);  add_83 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_171, view_225, permute_113);  primals_171 = None
    view_226: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_62, [1, 512, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_226, [1, 512, 16, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 16, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_116: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
    expand_40: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 16, 512, 64]);  permute_115 = None
    view_229: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_40, [16, 512, 64]);  expand_40 = None
    expand_41: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_116, [1, 16, 64, 512]);  permute_116 = None
    view_230: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_41, [16, 64, 512]);  expand_41 = None
    bmm_20: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
    view_231: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 16, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_84: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_84, [-1], True)
    sub_32: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_84, amax_10);  add_84 = amax_10 = None
    exp_10: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_104: "f32[1, 16, 512, 512]" = native_dropout_31[0]
    getitem_105: "b8[1, 16, 512, 512]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_42: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_104, [1, 16, 512, 512]);  getitem_104 = None
    view_232: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_42, [16, 512, 512]);  expand_42 = None
    expand_43: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_114, [1, 16, 512, 64]);  permute_114 = None
    view_233: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_43, [16, 512, 64]);  expand_43 = None
    bmm_21: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
    view_234: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 16, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_10: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_10, [1, 512, 1024]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 1024]" = torch.ops.aten.view.default(view_235, [512, 1024]);  view_235 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
    view_237: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_63, [1, 512, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 1024]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 1024]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_85: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_81, getitem_106);  getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_109)
    mul_73: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_73, primals_174);  mul_73 = None
    add_87: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_74, primals_175);  mul_74 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 1024]" = torch.ops.aten.view.default(add_87, [512, 1024]);  add_87 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_64: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
    view_239: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 512, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 4096]" = torch.ops.aten.view.default(mul_77, [512, 4096]);  mul_77 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 1024]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 1024]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_89: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_85, getitem_110);  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_113)
    mul_78: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, primals_180);  mul_78 = None
    add_91: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_79, primals_181);  mul_79 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 1024]" = torch.ops.aten.view.default(add_91, [512, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_66: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
    view_243: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_66, [1, 512, 1024]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_244: "f32[512, 1024]" = torch.ops.aten.view.default(add_91, [512, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_185, view_244, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_67, [1, 512, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 16, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_247: "f32[512, 1024]" = torch.ops.aten.view.default(add_91, [512, 1024]);  add_91 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_187, view_247, permute_124);  primals_187 = None
    view_248: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_68, [1, 512, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_248, [1, 512, 16, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 16, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_127: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_44: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 16, 512, 64]);  permute_126 = None
    view_251: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_44, [16, 512, 64]);  expand_44 = None
    expand_45: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_127, [1, 16, 64, 512]);  permute_127 = None
    view_252: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_45, [16, 64, 512]);  expand_45 = None
    bmm_22: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 16, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_92: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_35: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_92, amax_11);  add_92 = amax_11 = None
    exp_11: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_114: "f32[1, 16, 512, 512]" = native_dropout_34[0]
    getitem_115: "b8[1, 16, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_46: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_114, [1, 16, 512, 512]);  getitem_114 = None
    view_254: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_46, [16, 512, 512]);  expand_46 = None
    expand_47: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_125, [1, 16, 512, 64]);  permute_125 = None
    view_255: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_47, [16, 512, 64]);  expand_47 = None
    bmm_23: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 16, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_11: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_11, [1, 512, 1024]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 1024]" = torch.ops.aten.view.default(view_257, [512, 1024]);  view_257 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
    view_259: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_69, [1, 512, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 1024]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 1024]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_93: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_89, getitem_116);  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_119)
    mul_80: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_80, primals_190);  mul_80 = None
    add_95: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_81, primals_191);  mul_81 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 1024]" = torch.ops.aten.view.default(add_95, [512, 1024]);  add_95 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_70: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
    view_261: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 512, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 4096]" = torch.ops.aten.view.default(mul_84, [512, 4096]);  mul_84 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 1024]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 1024]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_97: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_93, getitem_120);  getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_123)
    mul_85: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, primals_196);  mul_85 = None
    add_99: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_86, primals_197);  mul_86 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_264: "f32[512, 1024]" = torch.ops.aten.view.default(add_99, [512, 1024])
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_72, [1, 512, 1024]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_266: "f32[512, 1024]" = torch.ops.aten.view.default(add_99, [512, 1024])
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_73: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_201, view_266, permute_133);  primals_201 = None
    view_267: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_73, [1, 512, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_268: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_267, [1, 512, 16, 64]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_134: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_269: "f32[512, 1024]" = torch.ops.aten.view.default(add_99, [512, 1024]);  add_99 = None
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm_74: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_203, view_269, permute_135);  primals_203 = None
    view_270: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_74, [1, 512, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_271: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_270, [1, 512, 16, 64]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_136: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_272: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_265, [1, 512, 16, 64]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_137: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_138: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_134, [0, 1, 3, 2]);  permute_134 = None
    expand_48: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_137, [1, 16, 512, 64]);  permute_137 = None
    view_273: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_48, [16, 512, 64]);  expand_48 = None
    expand_49: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_138, [1, 16, 64, 512]);  permute_138 = None
    view_274: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_49, [16, 64, 512]);  expand_49 = None
    bmm_24: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_273, view_274)
    view_275: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_24, [1, 16, 512, 512]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_24: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_275, 8.0);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_100: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_24, mul);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_12: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_100, [-1], True)
    sub_38: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_100, amax_12);  add_100 = amax_12 = None
    exp_12: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_25: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_37 = torch.ops.aten.native_dropout.default(div_25, 0.1, True);  div_25 = None
    getitem_124: "f32[1, 16, 512, 512]" = native_dropout_37[0]
    getitem_125: "b8[1, 16, 512, 512]" = native_dropout_37[1];  native_dropout_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_50: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_124, [1, 16, 512, 512]);  getitem_124 = None
    view_276: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_50, [16, 512, 512]);  expand_50 = None
    expand_51: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_136, [1, 16, 512, 64]);  permute_136 = None
    view_277: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_51, [16, 512, 64]);  expand_51 = None
    bmm_25: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_276, view_277)
    view_278: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_25, [1, 16, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    clone_12: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_279: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_12, [1, 512, 1024]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 1024]" = torch.ops.aten.view.default(view_279, [512, 1024]);  view_279 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_75: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_205, view_280, permute_140);  primals_205 = None
    view_281: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_75, [1, 512, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_38 = torch.ops.aten.native_dropout.default(view_281, 0.1, True);  view_281 = None
    getitem_126: "f32[1, 512, 1024]" = native_dropout_38[0]
    getitem_127: "b8[1, 512, 1024]" = native_dropout_38[1];  native_dropout_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_101: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_97, getitem_126);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
    getitem_128: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_129: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-12);  getitem_128 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_39: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_129)
    mul_87: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
    mul_88: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, primals_206);  mul_87 = None
    add_103: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_88, primals_207);  mul_88 = primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 1024]" = torch.ops.aten.view.default(add_103, [512, 1024]);  add_103 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
    addmm_76: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_209, view_282, permute_141);  primals_209 = None
    view_283: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_76, [1, 512, 4096]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_89: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
    mul_90: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
    erf_12: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_104: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_91: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_89, add_104);  mul_89 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 4096]" = torch.ops.aten.view.default(mul_91, [512, 4096]);  mul_91 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_77: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_211, view_284, permute_142);  primals_211 = None
    view_285: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_77, [1, 512, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_39 = torch.ops.aten.native_dropout.default(view_285, 0.1, True);  view_285 = None
    getitem_130: "f32[1, 512, 1024]" = native_dropout_39[0]
    getitem_131: "b8[1, 512, 1024]" = native_dropout_39[1];  native_dropout_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_105: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_101, getitem_130);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_132: "f32[1, 512, 1]" = var_mean_26[0]
    getitem_133: "f32[1, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-12);  getitem_132 = None
    rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_40: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_133)
    mul_92: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
    mul_93: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, primals_212);  mul_92 = None
    add_107: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_93, primals_213);  mul_93 = primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_286: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024])
    permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_78: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_215, view_286, permute_143);  primals_215 = None
    view_287: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_78, [1, 512, 1024]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_288: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024])
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_79: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_217, view_288, permute_144);  primals_217 = None
    view_289: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_79, [1, 512, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_290: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_289, [1, 512, 16, 64]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_145: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_291: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024]);  add_107 = None
    permute_146: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_80: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_219, view_291, permute_146);  primals_219 = None
    view_292: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_80, [1, 512, 1024]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_293: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_292, [1, 512, 16, 64]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_147: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_294: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_287, [1, 512, 16, 64]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_148: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_149: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_145, [0, 1, 3, 2]);  permute_145 = None
    expand_52: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_148, [1, 16, 512, 64]);  permute_148 = None
    view_295: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_52, [16, 512, 64]);  expand_52 = None
    expand_53: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_149, [1, 16, 64, 512]);  permute_149 = None
    view_296: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_53, [16, 64, 512]);  expand_53 = None
    bmm_26: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_295, view_296)
    view_297: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_26, [1, 16, 512, 512]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_26: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_297, 8.0);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_108: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_26, mul);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_13: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_108, [-1], True)
    sub_41: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_108, amax_13);  add_108 = amax_13 = None
    exp_13: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_14: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_27: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_40 = torch.ops.aten.native_dropout.default(div_27, 0.1, True);  div_27 = None
    getitem_134: "f32[1, 16, 512, 512]" = native_dropout_40[0]
    getitem_135: "b8[1, 16, 512, 512]" = native_dropout_40[1];  native_dropout_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_54: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_134, [1, 16, 512, 512]);  getitem_134 = None
    view_298: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_54, [16, 512, 512]);  expand_54 = None
    expand_55: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_147, [1, 16, 512, 64]);  permute_147 = None
    view_299: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_55, [16, 512, 64]);  expand_55 = None
    bmm_27: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_298, view_299)
    view_300: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 16, 512, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    clone_13: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_13, [1, 512, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[512, 1024]" = torch.ops.aten.view.default(view_301, [512, 1024]);  view_301 = None
    permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_81: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_221, view_302, permute_151);  primals_221 = None
    view_303: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_81, [1, 512, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_41 = torch.ops.aten.native_dropout.default(view_303, 0.1, True);  view_303 = None
    getitem_136: "f32[1, 512, 1024]" = native_dropout_41[0]
    getitem_137: "b8[1, 512, 1024]" = native_dropout_41[1];  native_dropout_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_109: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_105, getitem_136);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_138: "f32[1, 512, 1]" = var_mean_27[0]
    getitem_139: "f32[1, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_110: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-12);  getitem_138 = None
    rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_42: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_139)
    mul_94: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = None
    mul_95: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, primals_222);  mul_94 = None
    add_111: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_95, primals_223);  mul_95 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_304: "f32[512, 1024]" = torch.ops.aten.view.default(add_111, [512, 1024]);  add_111 = None
    permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    addmm_82: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_225, view_304, permute_152);  primals_225 = None
    view_305: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_82, [1, 512, 4096]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_96: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
    mul_97: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
    erf_13: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_112: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_98: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_112);  mul_96 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 4096]" = torch.ops.aten.view.default(mul_98, [512, 4096]);  mul_98 = None
    permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_83: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_227, view_306, permute_153);  primals_227 = None
    view_307: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_83, [1, 512, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_42 = torch.ops.aten.native_dropout.default(view_307, 0.1, True);  view_307 = None
    getitem_140: "f32[1, 512, 1024]" = native_dropout_42[0]
    getitem_141: "b8[1, 512, 1024]" = native_dropout_42[1];  native_dropout_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_113: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_109, getitem_140);  getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
    getitem_142: "f32[1, 512, 1]" = var_mean_28[0]
    getitem_143: "f32[1, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_114: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-12);  getitem_142 = None
    rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_43: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_113, getitem_143)
    mul_99: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
    mul_100: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, primals_228);  mul_99 = None
    add_115: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_100, primals_229);  mul_100 = primals_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_308: "f32[512, 1024]" = torch.ops.aten.view.default(add_115, [512, 1024])
    permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_84: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_231, view_308, permute_154);  primals_231 = None
    view_309: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_84, [1, 512, 1024]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_310: "f32[512, 1024]" = torch.ops.aten.view.default(add_115, [512, 1024])
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_85: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_233, view_310, permute_155);  primals_233 = None
    view_311: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_85, [1, 512, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_312: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_311, [1, 512, 16, 64]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_313: "f32[512, 1024]" = torch.ops.aten.view.default(add_115, [512, 1024]);  add_115 = None
    permute_157: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_86: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_235, view_313, permute_157);  primals_235 = None
    view_314: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_86, [1, 512, 1024]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_315: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_314, [1, 512, 16, 64]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_316: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_309, [1, 512, 16, 64]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_159: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_160: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_156, [0, 1, 3, 2]);  permute_156 = None
    expand_56: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_159, [1, 16, 512, 64]);  permute_159 = None
    view_317: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_56, [16, 512, 64]);  expand_56 = None
    expand_57: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_160, [1, 16, 64, 512]);  permute_160 = None
    view_318: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_57, [16, 64, 512]);  expand_57 = None
    bmm_28: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_317, view_318)
    view_319: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_28, [1, 16, 512, 512]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_28: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_319, 8.0);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_116: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_28, mul);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_14: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
    sub_44: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_116, amax_14);  add_116 = amax_14 = None
    exp_14: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_29: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_43 = torch.ops.aten.native_dropout.default(div_29, 0.1, True);  div_29 = None
    getitem_144: "f32[1, 16, 512, 512]" = native_dropout_43[0]
    getitem_145: "b8[1, 16, 512, 512]" = native_dropout_43[1];  native_dropout_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_58: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_144, [1, 16, 512, 512]);  getitem_144 = None
    view_320: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_58, [16, 512, 512]);  expand_58 = None
    expand_59: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_158, [1, 16, 512, 64]);  permute_158 = None
    view_321: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_59, [16, 512, 64]);  expand_59 = None
    bmm_29: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_320, view_321)
    view_322: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_29, [1, 16, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
    clone_14: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_323: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_14, [1, 512, 1024]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_324: "f32[512, 1024]" = torch.ops.aten.view.default(view_323, [512, 1024]);  view_323 = None
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    addmm_87: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_237, view_324, permute_162);  primals_237 = None
    view_325: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_87, [1, 512, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_44 = torch.ops.aten.native_dropout.default(view_325, 0.1, True);  view_325 = None
    getitem_146: "f32[1, 512, 1024]" = native_dropout_44[0]
    getitem_147: "b8[1, 512, 1024]" = native_dropout_44[1];  native_dropout_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_117: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_113, getitem_146);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_148: "f32[1, 512, 1]" = var_mean_29[0]
    getitem_149: "f32[1, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-12);  getitem_148 = None
    rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_45: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_149)
    mul_101: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = None
    mul_102: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_101, primals_238);  mul_101 = None
    add_119: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_102, primals_239);  mul_102 = primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_326: "f32[512, 1024]" = torch.ops.aten.view.default(add_119, [512, 1024]);  add_119 = None
    permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_88: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_241, view_326, permute_163);  primals_241 = None
    view_327: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_88, [1, 512, 4096]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
    mul_104: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_14: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_120: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_105: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_103, add_120);  mul_103 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 4096]" = torch.ops.aten.view.default(mul_105, [512, 4096]);  mul_105 = None
    permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_89: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_243, view_328, permute_164);  primals_243 = None
    view_329: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_89, [1, 512, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_45 = torch.ops.aten.native_dropout.default(view_329, 0.1, True);  view_329 = None
    getitem_150: "f32[1, 512, 1024]" = native_dropout_45[0]
    getitem_151: "b8[1, 512, 1024]" = native_dropout_45[1];  native_dropout_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_121: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_117, getitem_150);  getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1]" = var_mean_30[0]
    getitem_153: "f32[1, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-12);  getitem_152 = None
    rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_46: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_153)
    mul_106: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
    mul_107: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_106, primals_244);  mul_106 = None
    add_123: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_107, primals_245);  mul_107 = primals_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_330: "f32[512, 1024]" = torch.ops.aten.view.default(add_123, [512, 1024])
    permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_90: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_247, view_330, permute_165);  primals_247 = None
    view_331: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_90, [1, 512, 1024]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_332: "f32[512, 1024]" = torch.ops.aten.view.default(add_123, [512, 1024])
    permute_166: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_91: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_249, view_332, permute_166);  primals_249 = None
    view_333: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_91, [1, 512, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_334: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_333, [1, 512, 16, 64]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_335: "f32[512, 1024]" = torch.ops.aten.view.default(add_123, [512, 1024]);  add_123 = None
    permute_168: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_92: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_251, view_335, permute_168);  primals_251 = None
    view_336: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_92, [1, 512, 1024]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_337: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_336, [1, 512, 16, 64]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_169: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_338: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_331, [1, 512, 16, 64]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_170: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_171: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_167, [0, 1, 3, 2]);  permute_167 = None
    expand_60: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_170, [1, 16, 512, 64]);  permute_170 = None
    view_339: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_60, [16, 512, 64]);  expand_60 = None
    expand_61: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_171, [1, 16, 64, 512]);  permute_171 = None
    view_340: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_61, [16, 64, 512]);  expand_61 = None
    bmm_30: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_339, view_340)
    view_341: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_30, [1, 16, 512, 512]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_30: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_341, 8.0);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_124: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_30, mul);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_15: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_124, [-1], True)
    sub_47: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_124, amax_15);  add_124 = amax_15 = None
    exp_15: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_16: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_31: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_46 = torch.ops.aten.native_dropout.default(div_31, 0.1, True);  div_31 = None
    getitem_154: "f32[1, 16, 512, 512]" = native_dropout_46[0]
    getitem_155: "b8[1, 16, 512, 512]" = native_dropout_46[1];  native_dropout_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_62: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_154, [1, 16, 512, 512]);  getitem_154 = None
    view_342: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_62, [16, 512, 512]);  expand_62 = None
    expand_63: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_169, [1, 16, 512, 64]);  permute_169 = None
    view_343: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_63, [16, 512, 64]);  expand_63 = None
    bmm_31: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_342, view_343)
    view_344: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 16, 512, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    clone_15: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_345: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_15, [1, 512, 1024]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_346: "f32[512, 1024]" = torch.ops.aten.view.default(view_345, [512, 1024]);  view_345 = None
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    addmm_93: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_253, view_346, permute_173);  primals_253 = None
    view_347: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_93, [1, 512, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_47 = torch.ops.aten.native_dropout.default(view_347, 0.1, True);  view_347 = None
    getitem_156: "f32[1, 512, 1024]" = native_dropout_47[0]
    getitem_157: "b8[1, 512, 1024]" = native_dropout_47[1];  native_dropout_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_125: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_121, getitem_156);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_158: "f32[1, 512, 1]" = var_mean_31[0]
    getitem_159: "f32[1, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_126: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-12);  getitem_158 = None
    rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_48: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_159)
    mul_108: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = None
    mul_109: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, primals_254);  mul_108 = None
    add_127: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_109, primals_255);  mul_109 = primals_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[512, 1024]" = torch.ops.aten.view.default(add_127, [512, 1024]);  add_127 = None
    permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_94: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_257, view_348, permute_174);  primals_257 = None
    view_349: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_94, [1, 512, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_110: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
    mul_111: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
    erf_15: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_128: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_112: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, add_128);  mul_110 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_350: "f32[512, 4096]" = torch.ops.aten.view.default(mul_112, [512, 4096]);  mul_112 = None
    permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
    addmm_95: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_259, view_350, permute_175);  primals_259 = None
    view_351: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_95, [1, 512, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_48 = torch.ops.aten.native_dropout.default(view_351, 0.1, True);  view_351 = None
    getitem_160: "f32[1, 512, 1024]" = native_dropout_48[0]
    getitem_161: "b8[1, 512, 1024]" = native_dropout_48[1];  native_dropout_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_129: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_125, getitem_160);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_162: "f32[1, 512, 1]" = var_mean_32[0]
    getitem_163: "f32[1, 512, 1]" = var_mean_32[1];  var_mean_32 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-12);  getitem_162 = None
    rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_49: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_163)
    mul_113: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
    mul_114: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, primals_260);  mul_113 = None
    add_131: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_114, primals_261);  mul_114 = primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_352: "f32[512, 1024]" = torch.ops.aten.view.default(add_131, [512, 1024])
    permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_96: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_263, view_352, permute_176);  primals_263 = None
    view_353: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_96, [1, 512, 1024]);  addmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_354: "f32[512, 1024]" = torch.ops.aten.view.default(add_131, [512, 1024])
    permute_177: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    addmm_97: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_265, view_354, permute_177);  primals_265 = None
    view_355: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_97, [1, 512, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_356: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_355, [1, 512, 16, 64]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_178: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_357: "f32[512, 1024]" = torch.ops.aten.view.default(add_131, [512, 1024]);  add_131 = None
    permute_179: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    addmm_98: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_267, view_357, permute_179);  primals_267 = None
    view_358: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_98, [1, 512, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_359: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_358, [1, 512, 16, 64]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_180: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_360: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_353, [1, 512, 16, 64]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_181: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_182: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_178, [0, 1, 3, 2]);  permute_178 = None
    expand_64: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_181, [1, 16, 512, 64]);  permute_181 = None
    view_361: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_64, [16, 512, 64]);  expand_64 = None
    expand_65: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_182, [1, 16, 64, 512]);  permute_182 = None
    view_362: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_65, [16, 64, 512]);  expand_65 = None
    bmm_32: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_361, view_362)
    view_363: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_32, [1, 16, 512, 512]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_363, 8.0);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_132: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_32, mul);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_16: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_132, [-1], True)
    sub_50: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_132, amax_16);  add_132 = amax_16 = None
    exp_16: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_33: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_49 = torch.ops.aten.native_dropout.default(div_33, 0.1, True);  div_33 = None
    getitem_164: "f32[1, 16, 512, 512]" = native_dropout_49[0]
    getitem_165: "b8[1, 16, 512, 512]" = native_dropout_49[1];  native_dropout_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_66: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_164, [1, 16, 512, 512]);  getitem_164 = None
    view_364: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_66, [16, 512, 512]);  expand_66 = None
    expand_67: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_180, [1, 16, 512, 64]);  permute_180 = None
    view_365: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_67, [16, 512, 64]);  expand_67 = None
    bmm_33: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_364, view_365)
    view_366: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_33, [1, 16, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    clone_16: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_16, [1, 512, 1024]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_368: "f32[512, 1024]" = torch.ops.aten.view.default(view_367, [512, 1024]);  view_367 = None
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_99: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_269, view_368, permute_184);  primals_269 = None
    view_369: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_99, [1, 512, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_50 = torch.ops.aten.native_dropout.default(view_369, 0.1, True);  view_369 = None
    getitem_166: "f32[1, 512, 1024]" = native_dropout_50[0]
    getitem_167: "b8[1, 512, 1024]" = native_dropout_50[1];  native_dropout_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_133: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_129, getitem_166);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_168: "f32[1, 512, 1]" = var_mean_33[0]
    getitem_169: "f32[1, 512, 1]" = var_mean_33[1];  var_mean_33 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-12);  getitem_168 = None
    rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_51: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_133, getitem_169)
    mul_115: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = None
    mul_116: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, primals_270);  mul_115 = None
    add_135: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_116, primals_271);  mul_116 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_370: "f32[512, 1024]" = torch.ops.aten.view.default(add_135, [512, 1024]);  add_135 = None
    permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_100: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_273, view_370, permute_185);  primals_273 = None
    view_371: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_100, [1, 512, 4096]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_117: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
    mul_118: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_16: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_136: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_119: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_117, add_136);  mul_117 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_372: "f32[512, 4096]" = torch.ops.aten.view.default(mul_119, [512, 4096]);  mul_119 = None
    permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_101: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_275, view_372, permute_186);  primals_275 = None
    view_373: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_101, [1, 512, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_51 = torch.ops.aten.native_dropout.default(view_373, 0.1, True);  view_373 = None
    getitem_170: "f32[1, 512, 1024]" = native_dropout_51[0]
    getitem_171: "b8[1, 512, 1024]" = native_dropout_51[1];  native_dropout_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_137: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_133, getitem_170);  getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
    getitem_172: "f32[1, 512, 1]" = var_mean_34[0]
    getitem_173: "f32[1, 512, 1]" = var_mean_34[1];  var_mean_34 = None
    add_138: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-12);  getitem_172 = None
    rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_52: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_173)
    mul_120: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
    mul_121: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, primals_276);  mul_120 = None
    add_139: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_121, primals_277);  mul_121 = primals_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_374: "f32[512, 1024]" = torch.ops.aten.view.default(add_139, [512, 1024])
    permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_102: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_279, view_374, permute_187);  primals_279 = None
    view_375: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_102, [1, 512, 1024]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_376: "f32[512, 1024]" = torch.ops.aten.view.default(add_139, [512, 1024])
    permute_188: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    addmm_103: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_281, view_376, permute_188);  primals_281 = None
    view_377: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_103, [1, 512, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_378: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_377, [1, 512, 16, 64]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_379: "f32[512, 1024]" = torch.ops.aten.view.default(add_139, [512, 1024]);  add_139 = None
    permute_190: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    addmm_104: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_283, view_379, permute_190);  primals_283 = None
    view_380: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_104, [1, 512, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_381: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_380, [1, 512, 16, 64]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_191: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_382: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_375, [1, 512, 16, 64]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_192: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_193: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_189, [0, 1, 3, 2]);  permute_189 = None
    expand_68: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_192, [1, 16, 512, 64]);  permute_192 = None
    view_383: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_68, [16, 512, 64]);  expand_68 = None
    expand_69: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_193, [1, 16, 64, 512]);  permute_193 = None
    view_384: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_69, [16, 64, 512]);  expand_69 = None
    bmm_34: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_383, view_384)
    view_385: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_34, [1, 16, 512, 512]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_34: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_385, 8.0);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_140: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_34, mul);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_17: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_140, [-1], True)
    sub_53: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_140, amax_17);  add_140 = amax_17 = None
    exp_17: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_18: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_35: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_52 = torch.ops.aten.native_dropout.default(div_35, 0.1, True);  div_35 = None
    getitem_174: "f32[1, 16, 512, 512]" = native_dropout_52[0]
    getitem_175: "b8[1, 16, 512, 512]" = native_dropout_52[1];  native_dropout_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_70: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_174, [1, 16, 512, 512]);  getitem_174 = None
    view_386: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_70, [16, 512, 512]);  expand_70 = None
    expand_71: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_191, [1, 16, 512, 64]);  permute_191 = None
    view_387: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_71, [16, 512, 64]);  expand_71 = None
    bmm_35: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_386, view_387)
    view_388: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 16, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    clone_17: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_389: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_17, [1, 512, 1024]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 1024]" = torch.ops.aten.view.default(view_389, [512, 1024]);  view_389 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
    addmm_105: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_285, view_390, permute_195);  primals_285 = None
    view_391: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_105, [1, 512, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_53 = torch.ops.aten.native_dropout.default(view_391, 0.1, True);  view_391 = None
    getitem_176: "f32[1, 512, 1024]" = native_dropout_53[0]
    getitem_177: "b8[1, 512, 1024]" = native_dropout_53[1];  native_dropout_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_141: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, getitem_176);  getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_178: "f32[1, 512, 1]" = var_mean_35[0]
    getitem_179: "f32[1, 512, 1]" = var_mean_35[1];  var_mean_35 = None
    add_142: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-12);  getitem_178 = None
    rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_54: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_141, getitem_179)
    mul_122: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = None
    mul_123: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, primals_286);  mul_122 = None
    add_143: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_123, primals_287);  mul_123 = primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 1024]" = torch.ops.aten.view.default(add_143, [512, 1024]);  add_143 = None
    permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
    addmm_106: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_289, view_392, permute_196);  primals_289 = None
    view_393: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_106, [1, 512, 4096]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_124: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_125: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_17: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
    add_144: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_126: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_124, add_144);  mul_124 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 4096]" = torch.ops.aten.view.default(mul_126, [512, 4096]);  mul_126 = None
    permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
    addmm_107: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_291, view_394, permute_197);  primals_291 = None
    view_395: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_107, [1, 512, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_54 = torch.ops.aten.native_dropout.default(view_395, 0.1, True);  view_395 = None
    getitem_180: "f32[1, 512, 1024]" = native_dropout_54[0]
    getitem_181: "b8[1, 512, 1024]" = native_dropout_54[1];  native_dropout_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_145: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_141, getitem_180);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_182: "f32[1, 512, 1]" = var_mean_36[0]
    getitem_183: "f32[1, 512, 1]" = var_mean_36[1];  var_mean_36 = None
    add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-12);  getitem_182 = None
    rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_55: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_183)
    mul_127: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
    mul_128: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_127, primals_292);  mul_127 = None
    add_147: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_128, primals_293);  mul_128 = primals_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_396: "f32[512, 1024]" = torch.ops.aten.view.default(add_147, [512, 1024])
    permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_108: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_295, view_396, permute_198);  primals_295 = None
    view_397: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_108, [1, 512, 1024]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_398: "f32[512, 1024]" = torch.ops.aten.view.default(add_147, [512, 1024])
    permute_199: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
    addmm_109: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_297, view_398, permute_199);  primals_297 = None
    view_399: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_109, [1, 512, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_400: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_399, [1, 512, 16, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_401: "f32[512, 1024]" = torch.ops.aten.view.default(add_147, [512, 1024]);  add_147 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    addmm_110: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_299, view_401, permute_201);  primals_299 = None
    view_402: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_110, [1, 512, 1024]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_403: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_402, [1, 512, 16, 64]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_202: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_404: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 16, 64]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_203: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_204: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_200, [0, 1, 3, 2]);  permute_200 = None
    expand_72: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_203, [1, 16, 512, 64]);  permute_203 = None
    view_405: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_72, [16, 512, 64]);  expand_72 = None
    expand_73: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_204, [1, 16, 64, 512]);  permute_204 = None
    view_406: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_73, [16, 64, 512]);  expand_73 = None
    bmm_36: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_405, view_406)
    view_407: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_36, [1, 16, 512, 512]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_36: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_407, 8.0);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_148: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_36, mul);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_18: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_148, [-1], True)
    sub_56: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_148, amax_18);  add_148 = amax_18 = None
    exp_18: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_37: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_55 = torch.ops.aten.native_dropout.default(div_37, 0.1, True);  div_37 = None
    getitem_184: "f32[1, 16, 512, 512]" = native_dropout_55[0]
    getitem_185: "b8[1, 16, 512, 512]" = native_dropout_55[1];  native_dropout_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_74: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_184, [1, 16, 512, 512]);  getitem_184 = None
    view_408: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_74, [16, 512, 512]);  expand_74 = None
    expand_75: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_202, [1, 16, 512, 64]);  permute_202 = None
    view_409: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_75, [16, 512, 64]);  expand_75 = None
    bmm_37: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_408, view_409)
    view_410: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_37, [1, 16, 512, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_18: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_411: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_18, [1, 512, 1024]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 1024]" = torch.ops.aten.view.default(view_411, [512, 1024]);  view_411 = None
    permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_111: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_301, view_412, permute_206);  primals_301 = None
    view_413: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_111, [1, 512, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_56 = torch.ops.aten.native_dropout.default(view_413, 0.1, True);  view_413 = None
    getitem_186: "f32[1, 512, 1024]" = native_dropout_56[0]
    getitem_187: "b8[1, 512, 1024]" = native_dropout_56[1];  native_dropout_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_149: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_145, getitem_186);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
    getitem_188: "f32[1, 512, 1]" = var_mean_37[0]
    getitem_189: "f32[1, 512, 1]" = var_mean_37[1];  var_mean_37 = None
    add_150: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-12);  getitem_188 = None
    rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_57: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_189)
    mul_129: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = None
    mul_130: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_129, primals_302);  mul_129 = None
    add_151: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_130, primals_303);  mul_130 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(add_151, [512, 1024]);  add_151 = None
    permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm_112: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_305, view_414, permute_207);  primals_305 = None
    view_415: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_112, [1, 512, 4096]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_131: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_132: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_18: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_152: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_133: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_131, add_152);  mul_131 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_416: "f32[512, 4096]" = torch.ops.aten.view.default(mul_133, [512, 4096]);  mul_133 = None
    permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
    addmm_113: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_307, view_416, permute_208);  primals_307 = None
    view_417: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_113, [1, 512, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_57 = torch.ops.aten.native_dropout.default(view_417, 0.1, True);  view_417 = None
    getitem_190: "f32[1, 512, 1024]" = native_dropout_57[0]
    getitem_191: "b8[1, 512, 1024]" = native_dropout_57[1];  native_dropout_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_153: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_149, getitem_190);  getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_192: "f32[1, 512, 1]" = var_mean_38[0]
    getitem_193: "f32[1, 512, 1]" = var_mean_38[1];  var_mean_38 = None
    add_154: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-12);  getitem_192 = None
    rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_58: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_193)
    mul_134: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
    mul_135: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_134, primals_308);  mul_134 = None
    add_155: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_135, primals_309);  mul_135 = primals_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_418: "f32[512, 1024]" = torch.ops.aten.view.default(add_155, [512, 1024])
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
    addmm_114: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_311, view_418, permute_209);  primals_311 = None
    view_419: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_114, [1, 512, 1024]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_420: "f32[512, 1024]" = torch.ops.aten.view.default(add_155, [512, 1024])
    permute_210: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_312, [1, 0]);  primals_312 = None
    addmm_115: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_313, view_420, permute_210);  primals_313 = None
    view_421: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_115, [1, 512, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_422: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_421, [1, 512, 16, 64]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_211: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_423: "f32[512, 1024]" = torch.ops.aten.view.default(add_155, [512, 1024]);  add_155 = None
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_314, [1, 0]);  primals_314 = None
    addmm_116: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_315, view_423, permute_212);  primals_315 = None
    view_424: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_116, [1, 512, 1024]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_425: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_424, [1, 512, 16, 64]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_213: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_426: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 16, 64]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_214: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_215: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_211, [0, 1, 3, 2]);  permute_211 = None
    expand_76: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_214, [1, 16, 512, 64]);  permute_214 = None
    view_427: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_76, [16, 512, 64]);  expand_76 = None
    expand_77: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_215, [1, 16, 64, 512]);  permute_215 = None
    view_428: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_77, [16, 64, 512]);  expand_77 = None
    bmm_38: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_427, view_428)
    view_429: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_38, [1, 16, 512, 512]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_429, 8.0);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_156: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_38, mul);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_19: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_156, [-1], True)
    sub_59: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_156, amax_19);  add_156 = amax_19 = None
    exp_19: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_20: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_39: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_58 = torch.ops.aten.native_dropout.default(div_39, 0.1, True);  div_39 = None
    getitem_194: "f32[1, 16, 512, 512]" = native_dropout_58[0]
    getitem_195: "b8[1, 16, 512, 512]" = native_dropout_58[1];  native_dropout_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_78: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_194, [1, 16, 512, 512]);  getitem_194 = None
    view_430: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_78, [16, 512, 512]);  expand_78 = None
    expand_79: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_213, [1, 16, 512, 64]);  permute_213 = None
    view_431: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_79, [16, 512, 64]);  expand_79 = None
    bmm_39: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_430, view_431)
    view_432: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 16, 512, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    clone_19: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_433: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_19, [1, 512, 1024]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 1024]" = torch.ops.aten.view.default(view_433, [512, 1024]);  view_433 = None
    permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_316, [1, 0]);  primals_316 = None
    addmm_117: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_317, view_434, permute_217);  primals_317 = None
    view_435: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_117, [1, 512, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_59 = torch.ops.aten.native_dropout.default(view_435, 0.1, True);  view_435 = None
    getitem_196: "f32[1, 512, 1024]" = native_dropout_59[0]
    getitem_197: "b8[1, 512, 1024]" = native_dropout_59[1];  native_dropout_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_157: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_153, getitem_196);  getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_198: "f32[1, 512, 1]" = var_mean_39[0]
    getitem_199: "f32[1, 512, 1]" = var_mean_39[1];  var_mean_39 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-12);  getitem_198 = None
    rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_60: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_199)
    mul_136: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = None
    mul_137: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_136, primals_318);  mul_136 = None
    add_159: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_137, primals_319);  mul_137 = primals_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[512, 1024]" = torch.ops.aten.view.default(add_159, [512, 1024]);  add_159 = None
    permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_320, [1, 0]);  primals_320 = None
    addmm_118: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_321, view_436, permute_218);  primals_321 = None
    view_437: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_118, [1, 512, 4096]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_138: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
    mul_139: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
    erf_19: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_160: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_140: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_138, add_160);  mul_138 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_438: "f32[512, 4096]" = torch.ops.aten.view.default(mul_140, [512, 4096]);  mul_140 = None
    permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_322, [1, 0]);  primals_322 = None
    addmm_119: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_323, view_438, permute_219);  primals_323 = None
    view_439: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_119, [1, 512, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_60 = torch.ops.aten.native_dropout.default(view_439, 0.1, True);  view_439 = None
    getitem_200: "f32[1, 512, 1024]" = native_dropout_60[0]
    getitem_201: "b8[1, 512, 1024]" = native_dropout_60[1];  native_dropout_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_161: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_157, getitem_200);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_202: "f32[1, 512, 1]" = var_mean_40[0]
    getitem_203: "f32[1, 512, 1]" = var_mean_40[1];  var_mean_40 = None
    add_162: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-12);  getitem_202 = None
    rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_61: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_203)
    mul_141: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
    mul_142: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_141, primals_324);  mul_141 = None
    add_163: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_142, primals_325);  mul_142 = primals_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_440: "f32[512, 1024]" = torch.ops.aten.view.default(add_163, [512, 1024])
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_326, [1, 0]);  primals_326 = None
    addmm_120: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_327, view_440, permute_220);  primals_327 = None
    view_441: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_120, [1, 512, 1024]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_442: "f32[512, 1024]" = torch.ops.aten.view.default(add_163, [512, 1024])
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_328, [1, 0]);  primals_328 = None
    addmm_121: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_329, view_442, permute_221);  primals_329 = None
    view_443: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_121, [1, 512, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_444: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_443, [1, 512, 16, 64]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_445: "f32[512, 1024]" = torch.ops.aten.view.default(add_163, [512, 1024]);  add_163 = None
    permute_223: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_330, [1, 0]);  primals_330 = None
    addmm_122: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_331, view_445, permute_223);  primals_331 = None
    view_446: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_122, [1, 512, 1024]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_447: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_446, [1, 512, 16, 64]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_224: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_448: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_441, [1, 512, 16, 64]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_225: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_226: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_222, [0, 1, 3, 2]);  permute_222 = None
    expand_80: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_225, [1, 16, 512, 64]);  permute_225 = None
    view_449: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_80, [16, 512, 64]);  expand_80 = None
    expand_81: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_226, [1, 16, 64, 512]);  permute_226 = None
    view_450: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_81, [16, 64, 512]);  expand_81 = None
    bmm_40: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_449, view_450)
    view_451: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_40, [1, 16, 512, 512]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_40: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_451, 8.0);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_164: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_40, mul);  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_20: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_164, [-1], True)
    sub_62: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_164, amax_20);  add_164 = amax_20 = None
    exp_20: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
    sum_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_41: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_61 = torch.ops.aten.native_dropout.default(div_41, 0.1, True);  div_41 = None
    getitem_204: "f32[1, 16, 512, 512]" = native_dropout_61[0]
    getitem_205: "b8[1, 16, 512, 512]" = native_dropout_61[1];  native_dropout_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_82: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_204, [1, 16, 512, 512]);  getitem_204 = None
    view_452: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_82, [16, 512, 512]);  expand_82 = None
    expand_83: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_224, [1, 16, 512, 64]);  permute_224 = None
    view_453: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_83, [16, 512, 64]);  expand_83 = None
    bmm_41: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_452, view_453)
    view_454: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_41, [1, 16, 512, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    clone_20: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_455: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_20, [1, 512, 1024]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 1024]" = torch.ops.aten.view.default(view_455, [512, 1024]);  view_455 = None
    permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_332, [1, 0]);  primals_332 = None
    addmm_123: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_333, view_456, permute_228);  primals_333 = None
    view_457: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_123, [1, 512, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_62 = torch.ops.aten.native_dropout.default(view_457, 0.1, True);  view_457 = None
    getitem_206: "f32[1, 512, 1024]" = native_dropout_62[0]
    getitem_207: "b8[1, 512, 1024]" = native_dropout_62[1];  native_dropout_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_165: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_161, getitem_206);  getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
    getitem_208: "f32[1, 512, 1]" = var_mean_41[0]
    getitem_209: "f32[1, 512, 1]" = var_mean_41[1];  var_mean_41 = None
    add_166: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-12);  getitem_208 = None
    rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_63: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_209)
    mul_143: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = None
    mul_144: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_143, primals_334);  mul_143 = None
    add_167: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_144, primals_335);  mul_144 = primals_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_458: "f32[512, 1024]" = torch.ops.aten.view.default(add_167, [512, 1024]);  add_167 = None
    permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
    addmm_124: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_337, view_458, permute_229);  primals_337 = None
    view_459: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_124, [1, 512, 4096]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_145: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
    mul_146: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
    erf_20: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_168: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_147: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_145, add_168);  mul_145 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_460: "f32[512, 4096]" = torch.ops.aten.view.default(mul_147, [512, 4096]);  mul_147 = None
    permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
    addmm_125: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_339, view_460, permute_230);  primals_339 = None
    view_461: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_125, [1, 512, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_63 = torch.ops.aten.native_dropout.default(view_461, 0.1, True);  view_461 = None
    getitem_210: "f32[1, 512, 1024]" = native_dropout_63[0]
    getitem_211: "b8[1, 512, 1024]" = native_dropout_63[1];  native_dropout_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_169: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_165, getitem_210);  getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
    getitem_212: "f32[1, 512, 1]" = var_mean_42[0]
    getitem_213: "f32[1, 512, 1]" = var_mean_42[1];  var_mean_42 = None
    add_170: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-12);  getitem_212 = None
    rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_64: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_213)
    mul_148: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
    mul_149: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, primals_340);  mul_148 = None
    add_171: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_149, primals_341);  mul_149 = primals_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_462: "f32[512, 1024]" = torch.ops.aten.view.default(add_171, [512, 1024])
    permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_342, [1, 0]);  primals_342 = None
    addmm_126: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_343, view_462, permute_231);  primals_343 = None
    view_463: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_126, [1, 512, 1024]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_464: "f32[512, 1024]" = torch.ops.aten.view.default(add_171, [512, 1024])
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
    addmm_127: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_345, view_464, permute_232);  primals_345 = None
    view_465: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_127, [1, 512, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_465, [1, 512, 16, 64]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_233: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_467: "f32[512, 1024]" = torch.ops.aten.view.default(add_171, [512, 1024]);  add_171 = None
    permute_234: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_346, [1, 0]);  primals_346 = None
    addmm_128: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_347, view_467, permute_234);  primals_347 = None
    view_468: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_128, [1, 512, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_469: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_468, [1, 512, 16, 64]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_235: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_470: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_463, [1, 512, 16, 64]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_236: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_237: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_233, [0, 1, 3, 2]);  permute_233 = None
    expand_84: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_236, [1, 16, 512, 64]);  permute_236 = None
    view_471: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_84, [16, 512, 64]);  expand_84 = None
    expand_85: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_237, [1, 16, 64, 512]);  permute_237 = None
    view_472: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_85, [16, 64, 512]);  expand_85 = None
    bmm_42: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_471, view_472)
    view_473: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_42, [1, 16, 512, 512]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_42: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_473, 8.0);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_172: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_42, mul);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_21: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_172, [-1], True)
    sub_65: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_172, amax_21);  add_172 = amax_21 = None
    exp_21: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
    sum_22: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_43: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_64 = torch.ops.aten.native_dropout.default(div_43, 0.1, True);  div_43 = None
    getitem_214: "f32[1, 16, 512, 512]" = native_dropout_64[0]
    getitem_215: "b8[1, 16, 512, 512]" = native_dropout_64[1];  native_dropout_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_86: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_214, [1, 16, 512, 512]);  getitem_214 = None
    view_474: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_86, [16, 512, 512]);  expand_86 = None
    expand_87: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_235, [1, 16, 512, 64]);  permute_235 = None
    view_475: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_87, [16, 512, 64]);  expand_87 = None
    bmm_43: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_474, view_475)
    view_476: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 16, 512, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    clone_21: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_477: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_21, [1, 512, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_478: "f32[512, 1024]" = torch.ops.aten.view.default(view_477, [512, 1024]);  view_477 = None
    permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_348, [1, 0]);  primals_348 = None
    addmm_129: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_349, view_478, permute_239);  primals_349 = None
    view_479: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_129, [1, 512, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_65 = torch.ops.aten.native_dropout.default(view_479, 0.1, True);  view_479 = None
    getitem_216: "f32[1, 512, 1024]" = native_dropout_65[0]
    getitem_217: "b8[1, 512, 1024]" = native_dropout_65[1];  native_dropout_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_173: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_169, getitem_216);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
    getitem_218: "f32[1, 512, 1]" = var_mean_43[0]
    getitem_219: "f32[1, 512, 1]" = var_mean_43[1];  var_mean_43 = None
    add_174: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-12);  getitem_218 = None
    rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_66: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_219)
    mul_150: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = None
    mul_151: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, primals_350);  mul_150 = None
    add_175: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_151, primals_351);  mul_151 = primals_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_480: "f32[512, 1024]" = torch.ops.aten.view.default(add_175, [512, 1024]);  add_175 = None
    permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_352, [1, 0]);  primals_352 = None
    addmm_130: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_353, view_480, permute_240);  primals_353 = None
    view_481: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_130, [1, 512, 4096]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_152: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    mul_153: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
    erf_21: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_176: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_154: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_152, add_176);  mul_152 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_482: "f32[512, 4096]" = torch.ops.aten.view.default(mul_154, [512, 4096]);  mul_154 = None
    permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_354, [1, 0]);  primals_354 = None
    addmm_131: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_355, view_482, permute_241);  primals_355 = None
    view_483: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_131, [1, 512, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_66 = torch.ops.aten.native_dropout.default(view_483, 0.1, True);  view_483 = None
    getitem_220: "f32[1, 512, 1024]" = native_dropout_66[0]
    getitem_221: "b8[1, 512, 1024]" = native_dropout_66[1];  native_dropout_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_177: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_173, getitem_220);  getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
    getitem_222: "f32[1, 512, 1]" = var_mean_44[0]
    getitem_223: "f32[1, 512, 1]" = var_mean_44[1];  var_mean_44 = None
    add_178: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-12);  getitem_222 = None
    rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_67: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_223)
    mul_155: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
    mul_156: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_155, primals_356);  mul_155 = None
    add_179: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_156, primals_357);  mul_156 = primals_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_484: "f32[512, 1024]" = torch.ops.aten.view.default(add_179, [512, 1024])
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
    addmm_132: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_359, view_484, permute_242);  primals_359 = None
    view_485: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_132, [1, 512, 1024]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_486: "f32[512, 1024]" = torch.ops.aten.view.default(add_179, [512, 1024])
    permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None
    addmm_133: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_361, view_486, permute_243);  primals_361 = None
    view_487: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_133, [1, 512, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_488: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_487, [1, 512, 16, 64]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_244: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_489: "f32[512, 1024]" = torch.ops.aten.view.default(add_179, [512, 1024]);  add_179 = None
    permute_245: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_362, [1, 0]);  primals_362 = None
    addmm_134: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_363, view_489, permute_245);  primals_363 = None
    view_490: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_134, [1, 512, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_491: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_490, [1, 512, 16, 64]);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_246: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_492: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_485, [1, 512, 16, 64]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_247: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_248: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_244, [0, 1, 3, 2]);  permute_244 = None
    expand_88: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_247, [1, 16, 512, 64]);  permute_247 = None
    view_493: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_88, [16, 512, 64]);  expand_88 = None
    expand_89: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_248, [1, 16, 64, 512]);  permute_248 = None
    view_494: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_89, [16, 64, 512]);  expand_89 = None
    bmm_44: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_493, view_494)
    view_495: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_44, [1, 16, 512, 512]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_495, 8.0);  view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_180: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_44, mul);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_22: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_180, [-1], True)
    sub_68: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_180, amax_22);  add_180 = amax_22 = None
    exp_22: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_45: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_67 = torch.ops.aten.native_dropout.default(div_45, 0.1, True);  div_45 = None
    getitem_224: "f32[1, 16, 512, 512]" = native_dropout_67[0]
    getitem_225: "b8[1, 16, 512, 512]" = native_dropout_67[1];  native_dropout_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_90: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_224, [1, 16, 512, 512]);  getitem_224 = None
    view_496: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_90, [16, 512, 512]);  expand_90 = None
    expand_91: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_246, [1, 16, 512, 64]);  permute_246 = None
    view_497: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_91, [16, 512, 64]);  expand_91 = None
    bmm_45: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_496, view_497)
    view_498: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_45, [1, 16, 512, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    clone_22: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_499: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_22, [1, 512, 1024]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_500: "f32[512, 1024]" = torch.ops.aten.view.default(view_499, [512, 1024]);  view_499 = None
    permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None
    addmm_135: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_365, view_500, permute_250);  primals_365 = None
    view_501: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_135, [1, 512, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_68 = torch.ops.aten.native_dropout.default(view_501, 0.1, True);  view_501 = None
    getitem_226: "f32[1, 512, 1024]" = native_dropout_68[0]
    getitem_227: "b8[1, 512, 1024]" = native_dropout_68[1];  native_dropout_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_181: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_177, getitem_226);  getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
    getitem_228: "f32[1, 512, 1]" = var_mean_45[0]
    getitem_229: "f32[1, 512, 1]" = var_mean_45[1];  var_mean_45 = None
    add_182: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-12);  getitem_228 = None
    rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_69: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_181, getitem_229)
    mul_157: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = None
    mul_158: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, primals_366);  mul_157 = None
    add_183: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_158, primals_367);  mul_158 = primals_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_502: "f32[512, 1024]" = torch.ops.aten.view.default(add_183, [512, 1024]);  add_183 = None
    permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_368, [1, 0]);  primals_368 = None
    addmm_136: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_369, view_502, permute_251);  primals_369 = None
    view_503: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_136, [1, 512, 4096]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    mul_160: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
    erf_22: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_184: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_161: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_159, add_184);  mul_159 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 4096]" = torch.ops.aten.view.default(mul_161, [512, 4096]);  mul_161 = None
    permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_370, [1, 0]);  primals_370 = None
    addmm_137: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_371, view_504, permute_252);  primals_371 = None
    view_505: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_137, [1, 512, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_69 = torch.ops.aten.native_dropout.default(view_505, 0.1, True);  view_505 = None
    getitem_230: "f32[1, 512, 1024]" = native_dropout_69[0]
    getitem_231: "b8[1, 512, 1024]" = native_dropout_69[1];  native_dropout_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_185: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_181, getitem_230);  getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
    getitem_232: "f32[1, 512, 1]" = var_mean_46[0]
    getitem_233: "f32[1, 512, 1]" = var_mean_46[1];  var_mean_46 = None
    add_186: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-12);  getitem_232 = None
    rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_70: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_233)
    mul_162: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
    mul_163: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, primals_372);  mul_162 = None
    add_187: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_163, primals_373);  mul_163 = primals_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_506: "f32[512, 1024]" = torch.ops.aten.view.default(add_187, [512, 1024])
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_374, [1, 0]);  primals_374 = None
    addmm_138: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_375, view_506, permute_253);  primals_375 = None
    view_507: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_138, [1, 512, 1024]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_508: "f32[512, 1024]" = torch.ops.aten.view.default(add_187, [512, 1024])
    permute_254: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_376, [1, 0]);  primals_376 = None
    addmm_139: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_377, view_508, permute_254);  primals_377 = None
    view_509: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_139, [1, 512, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_510: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_509, [1, 512, 16, 64]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_511: "f32[512, 1024]" = torch.ops.aten.view.default(add_187, [512, 1024]);  add_187 = None
    permute_256: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_378, [1, 0]);  primals_378 = None
    addmm_140: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_379, view_511, permute_256);  primals_379 = None
    view_512: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_140, [1, 512, 1024]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_513: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_512, [1, 512, 16, 64]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_257: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_514: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_507, [1, 512, 16, 64]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_258: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_259: "f32[1, 16, 64, 512]" = torch.ops.aten.permute.default(permute_255, [0, 1, 3, 2]);  permute_255 = None
    expand_92: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_258, [1, 16, 512, 64]);  permute_258 = None
    view_515: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_92, [16, 512, 64]);  expand_92 = None
    expand_93: "f32[1, 16, 64, 512]" = torch.ops.aten.expand.default(permute_259, [1, 16, 64, 512]);  permute_259 = None
    view_516: "f32[16, 64, 512]" = torch.ops.aten.view.default(expand_93, [16, 64, 512]);  expand_93 = None
    bmm_46: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_515, view_516)
    view_517: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_46, [1, 16, 512, 512]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_46: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(view_517, 8.0);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:302, code: attention_scores = attention_scores + attention_mask
    add_188: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(div_46, mul);  div_46 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_23: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(add_188, [-1], True)
    sub_71: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_188, amax_23);  add_188 = amax_23 = None
    exp_23: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
    sum_24: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_47: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(div_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    native_dropout_70 = torch.ops.aten.native_dropout.default(div_47, 0.1, True);  div_47 = None
    getitem_234: "f32[1, 16, 512, 512]" = native_dropout_70[0]
    getitem_235: "b8[1, 16, 512, 512]" = native_dropout_70[1];  native_dropout_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_94: "f32[1, 16, 512, 512]" = torch.ops.aten.expand.default(getitem_234, [1, 16, 512, 512]);  getitem_234 = None
    view_518: "f32[16, 512, 512]" = torch.ops.aten.view.default(expand_94, [16, 512, 512]);  expand_94 = None
    expand_95: "f32[1, 16, 512, 64]" = torch.ops.aten.expand.default(permute_257, [1, 16, 512, 64]);  permute_257 = None
    view_519: "f32[16, 512, 64]" = torch.ops.aten.view.default(expand_95, [16, 512, 64]);  expand_95 = None
    bmm_47: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_518, view_519)
    view_520: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 16, 512, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    clone_23: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_521: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_23, [1, 512, 1024]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_522: "f32[512, 1024]" = torch.ops.aten.view.default(view_521, [512, 1024]);  view_521 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_380, [1, 0]);  primals_380 = None
    addmm_141: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_381, view_522, permute_261);  primals_381 = None
    view_523: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_141, [1, 512, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    native_dropout_71 = torch.ops.aten.native_dropout.default(view_523, 0.1, True);  view_523 = None
    getitem_236: "f32[1, 512, 1024]" = native_dropout_71[0]
    getitem_237: "b8[1, 512, 1024]" = native_dropout_71[1];  native_dropout_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:338, code: return residual + hidden_states
    add_189: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_185, getitem_236);  getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
    getitem_238: "f32[1, 512, 1]" = var_mean_47[0]
    getitem_239: "f32[1, 512, 1]" = var_mean_47[1];  var_mean_47 = None
    add_190: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-12);  getitem_238 = None
    rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_72: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_239)
    mul_164: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = None
    mul_165: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_164, primals_382);  mul_164 = None
    add_191: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_165, primals_383);  mul_165 = primals_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 1024]" = torch.ops.aten.view.default(add_191, [512, 1024]);  add_191 = None
    permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_384, [1, 0]);  primals_384 = None
    addmm_142: "f32[512, 4096]" = torch.ops.aten.addmm.default(primals_385, view_524, permute_262);  primals_385 = None
    view_525: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_142, [1, 512, 4096]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_166: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
    mul_167: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
    erf_23: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_192: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_168: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_166, add_192);  mul_166 = add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[512, 4096]" = torch.ops.aten.view.default(mul_168, [512, 4096]);  mul_168 = None
    permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_386, [1, 0]);  primals_386 = None
    addmm_143: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_387, view_526, permute_263);  primals_387 = None
    view_527: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_143, [1, 512, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    native_dropout_72 = torch.ops.aten.native_dropout.default(view_527, 0.1, True);  view_527 = None
    getitem_240: "f32[1, 512, 1024]" = native_dropout_72[0]
    getitem_241: "b8[1, 512, 1024]" = native_dropout_72[1];  native_dropout_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:419, code: return input_tensor + hidden_states
    add_193: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, getitem_240);  getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_242: "f32[1, 512, 1]" = var_mean_48[0]
    getitem_243: "f32[1, 512, 1]" = var_mean_48[1];  var_mean_48 = None
    add_194: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-12);  getitem_242 = None
    rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_73: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_243)
    mul_169: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = None
    mul_170: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_169, primals_388);  mul_169 = None
    add_195: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_170, primals_389);  mul_170 = primals_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:646, code: hidden_states = self.dense(hidden_states)
    view_528: "f32[512, 1024]" = torch.ops.aten.view.default(add_195, [512, 1024]);  add_195 = None
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_390, [1, 0]);  primals_390 = None
    addmm_144: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_391, view_528, permute_264);  primals_391 = None
    view_529: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_144, [1, 512, 1024]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_171: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
    mul_172: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
    erf_24: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_196: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_173: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_171, add_196);  mul_171 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:648, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(mul_173, [2], correction = 0, keepdim = True)
    getitem_244: "f32[1, 512, 1]" = var_mean_49[0]
    getitem_245: "f32[1, 512, 1]" = var_mean_49[1];  var_mean_49 = None
    add_197: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-12);  getitem_244 = None
    rsqrt_49: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_74: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_173, getitem_245)
    mul_174: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
    mul_175: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_174, primals_392);  mul_174 = None
    add_198: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_175, primals_393);  mul_175 = primals_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:669, code: hidden_states = self.decoder(hidden_states)
    view_530: "f32[512, 1024]" = torch.ops.aten.view.default(add_198, [512, 1024]);  add_198 = None
    permute_265: "f32[1024, 29056]" = torch.ops.aten.permute.default(primals_394, [1, 0]);  primals_394 = None
    addmm_145: "f32[512, 29056]" = torch.ops.aten.addmm.default(primals_395, view_530, permute_265);  primals_395 = None
    view_531: "f32[1, 512, 29056]" = torch.ops.aten.view.default(addmm_145, [1, 512, 29056]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1230, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    slice_4: "f32[1, 512, 29056]" = torch.ops.aten.slice.Tensor(view_531, 0, 0, 9223372036854775807)
    slice_5: "f32[1, 511, 29056]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, -1);  slice_4 = None
    slice_6: "f32[1, 511, 29056]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1231, code: labels = labels[:, 1:].contiguous()
    slice_7: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_397, 0, 0, 9223372036854775807);  primals_397 = None
    slice_8: "i64[1, 511]" = torch.ops.aten.slice.Tensor(slice_7, 1, 1, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1233, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_532: "f32[511, 29056]" = torch.ops.aten.view.default(slice_6, [-1, 29056]);  slice_6 = None
    view_533: "i64[511]" = torch.ops.aten.view.default(slice_8, [-1]);  slice_8 = None
    amax_24: "f32[511, 1]" = torch.ops.aten.amax.default(view_532, [1], True)
    sub_75: "f32[511, 29056]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
    exp_24: "f32[511, 29056]" = torch.ops.aten.exp.default(sub_75)
    sum_25: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_76: "f32[511, 29056]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
    alias_24: "f32[511, 29056]" = torch.ops.aten.alias.default(sub_76)
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_533, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[511]" = torch.ops.aten.where.self(ne, view_533, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_2);  sub_76 = unsqueeze_2 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[511]" = torch.ops.aten.ne.Scalar(view_533, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[511]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[511]" = torch.ops.aten.ne.Scalar(view_533, -100)
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_3: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
    ne_3: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[511, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_2: "f32[511, 29056]" = torch.ops.aten.full.default([511, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[511, 29056]" = torch.ops.aten.scatter.value(full_2, 1, where_2, -1.0);  full_2 = where_2 = None
    ne_4: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100);  unsqueeze_3 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[511, 1]" = torch.ops.aten.where.self(ne_4, div_49, scalar_tensor_3);  ne_4 = div_49 = scalar_tensor_3 = None
    mul_176: "f32[511, 29056]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_25: "f32[511, 29056]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_25: "f32[511, 29056]" = torch.ops.aten.exp.default(alias_25);  alias_25 = None
    sum_28: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [1], True)
    mul_177: "f32[511, 29056]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_77: "f32[511, 29056]" = torch.ops.aten.sub.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    view_534: "f32[1, 511, 29056]" = torch.ops.aten.view.default(sub_77, [1, 511, 29056]);  sub_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1230, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    full_3: "f32[1, 511, 29056]" = torch.ops.aten.full.default([1, 511, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[1, 511, 29056]" = torch.ops.aten.slice_scatter.default(full_3, view_534, 2, 0, 9223372036854775807);  full_3 = view_534 = None
    full_4: "f32[1, 512, 29056]" = torch.ops.aten.full.default([1, 512, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 29056]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter, 1, 0, -1);  full_4 = slice_scatter = None
    full_5: "f32[1, 512, 29056]" = torch.ops.aten.full.default([1, 512, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[1, 512, 29056]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_1, 0, 0, 9223372036854775807);  full_5 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1230, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    add_199: "f32[1, 512, 29056]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_2);  tangents_2 = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:669, code: hidden_states = self.decoder(hidden_states)
    view_535: "f32[512, 29056]" = torch.ops.aten.view.default(add_199, [512, 29056]);  add_199 = None
    permute_266: "f32[29056, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    mm: "f32[512, 1024]" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
    permute_267: "f32[29056, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_1: "f32[29056, 1024]" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
    permute_268: "f32[1024, 29056]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_29: "f32[1, 29056]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[29056]" = torch.ops.aten.view.default(sum_29, [29056]);  sum_29 = None
    permute_269: "f32[29056, 1024]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_537: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm, [1, 512, 1024]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:648, code: hidden_states = self.LayerNorm(hidden_states)
    sub_78: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_173, getitem_245);  mul_173 = getitem_245 = None
    mul_178: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_49);  sub_78 = None
    mul_179: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, primals_392);  primals_392 = None
    mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_179, 1024)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_179, mul_178);  mul_179 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_178, sum_31);  sum_31 = None
    sub_79: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_180, sum_30);  mul_180 = sum_30 = None
    sub_80: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_79, mul_182);  sub_79 = mul_182 = None
    div_50: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
    mul_183: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_80);  div_50 = sub_80 = None
    mul_184: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, mul_178);  mul_178 = None
    sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_185: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
    erf_25: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
    add_200: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_186: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_200, 0.5);  add_200 = None
    mul_187: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, view_529)
    mul_188: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_187, -0.5);  mul_187 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_188);  mul_188 = None
    mul_189: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_190: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, mul_189);  view_529 = mul_189 = None
    add_201: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_186, mul_190);  mul_186 = mul_190 = None
    mul_191: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_183, add_201);  mul_183 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:646, code: hidden_states = self.dense(hidden_states)
    view_538: "f32[512, 1024]" = torch.ops.aten.view.default(mul_191, [512, 1024]);  mul_191 = None
    permute_270: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    mm_2: "f32[512, 1024]" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
    permute_271: "f32[1024, 512]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_3: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_34: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
    view_539: "f32[1024]" = torch.ops.aten.view.default(sum_34, [1024]);  sum_34 = None
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_540: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_2, [1, 512, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    sub_81: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_243);  add_193 = getitem_243 = None
    mul_192: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_48);  sub_81 = None
    mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, primals_388);  primals_388 = None
    mul_194: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_193, 1024)
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
    mul_195: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_193, mul_192);  mul_193 = None
    sum_36: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    mul_196: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_192, sum_36);  sum_36 = None
    sub_82: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_194, sum_35);  mul_194 = sum_35 = None
    sub_83: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_82, mul_196);  sub_82 = mul_196 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
    mul_197: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_83);  div_51 = sub_83 = None
    mul_198: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, mul_192);  mul_192 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
    sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None
    mul_199: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_200: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_197, mul_199);  mul_199 = None
    clone_24: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_200, memory_format = torch.contiguous_format);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_541: "f32[512, 1024]" = torch.ops.aten.view.default(clone_24, [512, 1024]);  clone_24 = None
    permute_274: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    mm_4: "f32[512, 4096]" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
    permute_275: "f32[1024, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_5: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
    permute_276: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_39: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[1024]" = torch.ops.aten.view.default(sum_39, [1024]);  sum_39 = None
    permute_277: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_543: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_4, [1, 512, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_201: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
    erf_26: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_202: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_202: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_202, 0.5);  add_202 = None
    mul_203: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, view_525)
    mul_204: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
    exp_27: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_204);  mul_204 = None
    mul_205: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_206: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, mul_205);  view_525 = mul_205 = None
    add_203: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
    mul_207: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_543, add_203);  view_543 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_544: "f32[512, 4096]" = torch.ops.aten.view.default(mul_207, [512, 4096]);  mul_207 = None
    permute_278: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    mm_6: "f32[512, 1024]" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
    permute_279: "f32[4096, 512]" = torch.ops.aten.permute.default(view_544, [1, 0])
    mm_7: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
    permute_280: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_40: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
    view_545: "f32[4096]" = torch.ops.aten.view.default(sum_40, [4096]);  sum_40 = None
    permute_281: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_546: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_6, [1, 512, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_84: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_189, getitem_239);  add_189 = getitem_239 = None
    mul_208: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_47);  sub_84 = None
    mul_209: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, primals_382);  primals_382 = None
    mul_210: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_209, 1024)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
    mul_211: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_209, mul_208);  mul_209 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    mul_212: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_208, sum_42);  sum_42 = None
    sub_85: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_210, sum_41);  mul_210 = sum_41 = None
    sub_86: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_85, mul_212);  sub_85 = mul_212 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None
    mul_213: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_86);  div_52 = sub_86 = None
    mul_214: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, mul_208);  mul_208 = None
    sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_546, [0, 1]);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_204: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_197, mul_213);  mul_197 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
    mul_215: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_216: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_204, mul_215);  mul_215 = None
    clone_25: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_216, memory_format = torch.contiguous_format);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_547: "f32[512, 1024]" = torch.ops.aten.view.default(clone_25, [512, 1024]);  clone_25 = None
    permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    mm_8: "f32[512, 1024]" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
    permute_283: "f32[1024, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_45: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[1024]" = torch.ops.aten.view.default(sum_45, [1024]);  sum_45 = None
    permute_285: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_549: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_8, [1, 512, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_550: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_549, [1, 512, 16, 64]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_551: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_286, [16, 512, 64]);  permute_286 = None
    permute_287: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
    bmm_48: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_551);  permute_287 = None
    permute_288: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
    bmm_49: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_551, permute_288);  view_551 = permute_288 = None
    view_552: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 16, 512, 64]);  bmm_48 = None
    view_553: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 16, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_3: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_235, torch.float32);  getitem_235 = None
    mul_217: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_218: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_553, mul_217);  view_553 = mul_217 = None
    clone_26: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_218, memory_format = torch.contiguous_format);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_26: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_219: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_26, alias_26);  clone_26 = None
    sum_46: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_46);  alias_26 = sum_46 = None
    sub_87: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_87, 8.0);  sub_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_554: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_53, [16, 512, 512]);  div_53 = None
    permute_289: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
    bmm_50: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_554);  permute_289 = None
    permute_290: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None
    bmm_51: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_554, permute_290);  view_554 = permute_290 = None
    view_555: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 16, 64, 512]);  bmm_50 = None
    view_556: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 16, 512, 64]);  bmm_51 = None
    permute_291: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_555, [0, 1, 3, 2]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_557: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_27, [1, 512, 1024]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_28: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_558: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_28, [1, 512, 1024]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_559: "f32[512, 1024]" = torch.ops.aten.view.default(view_558, [512, 1024]);  view_558 = None
    permute_294: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    mm_10: "f32[512, 1024]" = torch.ops.aten.mm.default(view_559, permute_294);  permute_294 = None
    permute_295: "f32[1024, 512]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_295, view_511);  permute_295 = view_511 = None
    permute_296: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[1024]" = torch.ops.aten.view.default(sum_47, [1024]);  sum_47 = None
    permute_297: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_561: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_10, [1, 512, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_562: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_298, [1, 512, 1024]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_563: "f32[512, 1024]" = torch.ops.aten.view.default(view_562, [512, 1024]);  view_562 = None
    permute_299: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    mm_12: "f32[512, 1024]" = torch.ops.aten.mm.default(view_563, permute_299);  permute_299 = None
    permute_300: "f32[1024, 512]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_300, view_508);  permute_300 = view_508 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[1024]" = torch.ops.aten.view.default(sum_48, [1024]);  sum_48 = None
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_565: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_12, [1, 512, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_205: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_561, view_565);  view_561 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_566: "f32[512, 1024]" = torch.ops.aten.view.default(view_557, [512, 1024]);  view_557 = None
    permute_303: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    mm_14: "f32[512, 1024]" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
    permute_304: "f32[1024, 512]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
    permute_305: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_49: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
    view_567: "f32[1024]" = torch.ops.aten.view.default(sum_49, [1024]);  sum_49 = None
    permute_306: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_568: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_14, [1, 512, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_206: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_205, view_568);  add_205 = view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_88: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_233);  add_185 = getitem_233 = None
    mul_221: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_46);  sub_88 = None
    mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, primals_372);  primals_372 = None
    mul_223: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_222, 1024)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_222, mul_221);  mul_222 = None
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_221, sum_51);  sum_51 = None
    sub_89: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_223, sum_50);  mul_223 = sum_50 = None
    sub_90: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_89, mul_225);  sub_89 = mul_225 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None
    mul_226: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_90);  div_54 = sub_90 = None
    mul_227: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, mul_221);  mul_221 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_207: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_204, mul_226);  add_204 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
    mul_228: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_229: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_207, mul_228);  mul_228 = None
    clone_29: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_229, memory_format = torch.contiguous_format);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_569: "f32[512, 1024]" = torch.ops.aten.view.default(clone_29, [512, 1024]);  clone_29 = None
    permute_307: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    mm_16: "f32[512, 4096]" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
    permute_308: "f32[1024, 512]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_17: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
    permute_309: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_54: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[1024]" = torch.ops.aten.view.default(sum_54, [1024]);  sum_54 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_571: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_16, [1, 512, 4096]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_230: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
    erf_27: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_208: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_231: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_232: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, view_503)
    mul_233: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
    exp_28: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
    mul_234: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_235: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, mul_234);  view_503 = mul_234 = None
    add_209: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
    mul_236: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_571, add_209);  view_571 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_572: "f32[512, 4096]" = torch.ops.aten.view.default(mul_236, [512, 4096]);  mul_236 = None
    permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    mm_18: "f32[512, 1024]" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
    permute_312: "f32[4096, 512]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_19: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
    permute_313: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_55: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[4096]" = torch.ops.aten.view.default(sum_55, [4096]);  sum_55 = None
    permute_314: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_574: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_18, [1, 512, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_91: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_181, getitem_229);  add_181 = getitem_229 = None
    mul_237: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_45);  sub_91 = None
    mul_238: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, primals_366);  primals_366 = None
    mul_239: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_238, 1024)
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_238, mul_237);  mul_238 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_237, sum_57);  sum_57 = None
    sub_92: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_239, sum_56);  mul_239 = sum_56 = None
    sub_93: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_92, mul_241);  sub_92 = mul_241 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None
    mul_242: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_93);  div_55 = sub_93 = None
    mul_243: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, mul_237);  mul_237 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_210: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_207, mul_242);  add_207 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None
    mul_244: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_245: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_210, mul_244);  mul_244 = None
    clone_30: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_245, memory_format = torch.contiguous_format);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_575: "f32[512, 1024]" = torch.ops.aten.view.default(clone_30, [512, 1024]);  clone_30 = None
    permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    mm_20: "f32[512, 1024]" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
    permute_316: "f32[1024, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_21: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
    permute_317: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[1024]" = torch.ops.aten.view.default(sum_60, [1024]);  sum_60 = None
    permute_318: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_577: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_20, [1, 512, 1024]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_578: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_577, [1, 512, 16, 64]);  view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_579: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_319, [16, 512, 64]);  permute_319 = None
    permute_320: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
    bmm_52: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_579);  permute_320 = None
    permute_321: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
    bmm_53: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_579, permute_321);  view_579 = permute_321 = None
    view_580: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 16, 512, 64]);  bmm_52 = None
    view_581: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 16, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_6: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
    mul_246: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_247: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_581, mul_246);  view_581 = mul_246 = None
    clone_31: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_247, memory_format = torch.contiguous_format);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_27: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_248: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_31, alias_27);  clone_31 = None
    sum_61: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [-1], True)
    mul_249: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_61);  alias_27 = sum_61 = None
    sub_94: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_94, 8.0);  sub_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_582: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_56, [16, 512, 512]);  div_56 = None
    permute_322: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
    bmm_54: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_582);  permute_322 = None
    permute_323: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
    bmm_55: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_582, permute_323);  view_582 = permute_323 = None
    view_583: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 16, 64, 512]);  bmm_54 = None
    view_584: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 16, 512, 64]);  bmm_55 = None
    permute_324: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_583, [0, 1, 3, 2]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_585: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_32, [1, 512, 1024]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_33: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_586: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_33, [1, 512, 1024]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_587: "f32[512, 1024]" = torch.ops.aten.view.default(view_586, [512, 1024]);  view_586 = None
    permute_327: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
    mm_22: "f32[512, 1024]" = torch.ops.aten.mm.default(view_587, permute_327);  permute_327 = None
    permute_328: "f32[1024, 512]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_23: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_328, view_489);  permute_328 = view_489 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[1024]" = torch.ops.aten.view.default(sum_62, [1024]);  sum_62 = None
    permute_330: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_589: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_22, [1, 512, 1024]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_590: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_331, [1, 512, 1024]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_591: "f32[512, 1024]" = torch.ops.aten.view.default(view_590, [512, 1024]);  view_590 = None
    permute_332: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    mm_24: "f32[512, 1024]" = torch.ops.aten.mm.default(view_591, permute_332);  permute_332 = None
    permute_333: "f32[1024, 512]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_333, view_486);  permute_333 = view_486 = None
    permute_334: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[1024]" = torch.ops.aten.view.default(sum_63, [1024]);  sum_63 = None
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_593: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_24, [1, 512, 1024]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_211: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_589, view_593);  view_589 = view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_594: "f32[512, 1024]" = torch.ops.aten.view.default(view_585, [512, 1024]);  view_585 = None
    permute_336: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    mm_26: "f32[512, 1024]" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
    permute_337: "f32[1024, 512]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_27: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
    permute_338: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_64: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[1024]" = torch.ops.aten.view.default(sum_64, [1024]);  sum_64 = None
    permute_339: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_596: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_26, [1, 512, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_212: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_211, view_596);  add_211 = view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_95: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_223);  add_177 = getitem_223 = None
    mul_250: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_44);  sub_95 = None
    mul_251: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, primals_356);  primals_356 = None
    mul_252: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_251, 1024)
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_251, mul_250);  mul_251 = None
    sum_66: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, sum_66);  sum_66 = None
    sub_96: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_252, sum_65);  mul_252 = sum_65 = None
    sub_97: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_96, mul_254);  sub_96 = mul_254 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None
    mul_255: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_97);  div_57 = sub_97 = None
    mul_256: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, mul_250);  mul_250 = None
    sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_213: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_210, mul_255);  add_210 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
    mul_257: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_258: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_213, mul_257);  mul_257 = None
    clone_34: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_258, memory_format = torch.contiguous_format);  mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_597: "f32[512, 1024]" = torch.ops.aten.view.default(clone_34, [512, 1024]);  clone_34 = None
    permute_340: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    mm_28: "f32[512, 4096]" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
    permute_341: "f32[1024, 512]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_29: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
    permute_342: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_69: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[1024]" = torch.ops.aten.view.default(sum_69, [1024]);  sum_69 = None
    permute_343: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_599: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_28, [1, 512, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_259: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
    erf_28: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
    add_214: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_260: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_261: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, view_481)
    mul_262: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
    exp_29: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
    mul_263: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_264: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, mul_263);  view_481 = mul_263 = None
    add_215: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
    mul_265: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_599, add_215);  view_599 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_600: "f32[512, 4096]" = torch.ops.aten.view.default(mul_265, [512, 4096]);  mul_265 = None
    permute_344: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    mm_30: "f32[512, 1024]" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
    permute_345: "f32[4096, 512]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_31: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
    permute_346: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[4096]" = torch.ops.aten.view.default(sum_70, [4096]);  sum_70 = None
    permute_347: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_602: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_30, [1, 512, 1024]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_98: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_219);  add_173 = getitem_219 = None
    mul_266: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_43);  sub_98 = None
    mul_267: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, primals_350);  primals_350 = None
    mul_268: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_267, 1024)
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_267, mul_266);  mul_267 = None
    sum_72: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_266, sum_72);  sum_72 = None
    sub_99: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_268, sum_71);  mul_268 = sum_71 = None
    sub_100: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_99, mul_270);  sub_99 = mul_270 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None
    mul_271: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_100);  div_58 = sub_100 = None
    mul_272: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, mul_266);  mul_266 = None
    sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_602, [0, 1]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_216: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_213, mul_271);  add_213 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None
    mul_273: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_274: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_216, mul_273);  mul_273 = None
    clone_35: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_274, memory_format = torch.contiguous_format);  mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_603: "f32[512, 1024]" = torch.ops.aten.view.default(clone_35, [512, 1024]);  clone_35 = None
    permute_348: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    mm_32: "f32[512, 1024]" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
    permute_349: "f32[1024, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
    permute_350: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[1024]" = torch.ops.aten.view.default(sum_75, [1024]);  sum_75 = None
    permute_351: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_605: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_32, [1, 512, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_606: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_605, [1, 512, 16, 64]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_607: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_352, [16, 512, 64]);  permute_352 = None
    permute_353: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
    bmm_56: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_607);  permute_353 = None
    permute_354: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
    bmm_57: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_607, permute_354);  view_607 = permute_354 = None
    view_608: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 16, 512, 64]);  bmm_56 = None
    view_609: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 16, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_9: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None
    mul_275: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_276: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_609, mul_275);  view_609 = mul_275 = None
    clone_36: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_276, memory_format = torch.contiguous_format);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_28: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_277: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_36, alias_28);  clone_36 = None
    sum_76: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
    mul_278: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_28, sum_76);  alias_28 = sum_76 = None
    sub_101: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_101, 8.0);  sub_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_610: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_59, [16, 512, 512]);  div_59 = None
    permute_355: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
    bmm_58: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_610);  permute_355 = None
    permute_356: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
    bmm_59: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_610, permute_356);  view_610 = permute_356 = None
    view_611: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 16, 64, 512]);  bmm_58 = None
    view_612: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 16, 512, 64]);  bmm_59 = None
    permute_357: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_611, [0, 1, 3, 2]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_613: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_37, [1, 512, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_38: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_614: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_38, [1, 512, 1024]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_615: "f32[512, 1024]" = torch.ops.aten.view.default(view_614, [512, 1024]);  view_614 = None
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    mm_34: "f32[512, 1024]" = torch.ops.aten.mm.default(view_615, permute_360);  permute_360 = None
    permute_361: "f32[1024, 512]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_35: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_361, view_467);  permute_361 = view_467 = None
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[1024]" = torch.ops.aten.view.default(sum_77, [1024]);  sum_77 = None
    permute_363: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_617: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_34, [1, 512, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_618: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_364, [1, 512, 1024]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_619: "f32[512, 1024]" = torch.ops.aten.view.default(view_618, [512, 1024]);  view_618 = None
    permute_365: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    mm_36: "f32[512, 1024]" = torch.ops.aten.mm.default(view_619, permute_365);  permute_365 = None
    permute_366: "f32[1024, 512]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_37: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_366, view_464);  permute_366 = view_464 = None
    permute_367: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
    view_620: "f32[1024]" = torch.ops.aten.view.default(sum_78, [1024]);  sum_78 = None
    permute_368: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_621: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_36, [1, 512, 1024]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_217: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_617, view_621);  view_617 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_622: "f32[512, 1024]" = torch.ops.aten.view.default(view_613, [512, 1024]);  view_613 = None
    permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    mm_38: "f32[512, 1024]" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
    permute_370: "f32[1024, 512]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_39: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
    permute_371: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_79: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
    view_623: "f32[1024]" = torch.ops.aten.view.default(sum_79, [1024]);  sum_79 = None
    permute_372: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_624: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_38, [1, 512, 1024]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_218: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_217, view_624);  add_217 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_102: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_213);  add_169 = getitem_213 = None
    mul_279: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_42);  sub_102 = None
    mul_280: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, primals_340);  primals_340 = None
    mul_281: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, 1024)
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, mul_279);  mul_280 = None
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_279, sum_81);  sum_81 = None
    sub_103: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_281, sum_80);  mul_281 = sum_80 = None
    sub_104: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_103, mul_283);  sub_103 = mul_283 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None
    mul_284: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_104);  div_60 = sub_104 = None
    mul_285: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, mul_279);  mul_279 = None
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_219: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_216, mul_284);  add_216 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None
    mul_286: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_287: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_219, mul_286);  mul_286 = None
    clone_39: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_287, memory_format = torch.contiguous_format);  mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_625: "f32[512, 1024]" = torch.ops.aten.view.default(clone_39, [512, 1024]);  clone_39 = None
    permute_373: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    mm_40: "f32[512, 4096]" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
    permute_374: "f32[1024, 512]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_41: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
    permute_375: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
    view_626: "f32[1024]" = torch.ops.aten.view.default(sum_84, [1024]);  sum_84 = None
    permute_376: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_627: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_40, [1, 512, 4096]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_288: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
    erf_29: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_288);  mul_288 = None
    add_220: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_289: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_290: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, view_459)
    mul_291: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_30: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_293: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, mul_292);  view_459 = mul_292 = None
    add_221: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_627, add_221);  view_627 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[512, 4096]" = torch.ops.aten.view.default(mul_294, [512, 4096]);  mul_294 = None
    permute_377: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    mm_42: "f32[512, 1024]" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
    permute_378: "f32[4096, 512]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_43: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
    permute_379: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_85: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[4096]" = torch.ops.aten.view.default(sum_85, [4096]);  sum_85 = None
    permute_380: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_630: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_42, [1, 512, 1024]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_105: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_165, getitem_209);  add_165 = getitem_209 = None
    mul_295: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_41);  sub_105 = None
    mul_296: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, primals_334);  primals_334 = None
    mul_297: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_296, 1024)
    sum_86: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
    mul_298: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_296, mul_295);  mul_296 = None
    sum_87: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_295, sum_87);  sum_87 = None
    sub_106: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_297, sum_86);  mul_297 = sum_86 = None
    sub_107: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_106, mul_299);  sub_106 = mul_299 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None
    mul_300: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_107);  div_61 = sub_107 = None
    mul_301: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, mul_295);  mul_295 = None
    sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
    sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_630, [0, 1]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_222: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_219, mul_300);  add_219 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
    mul_302: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_303: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_222, mul_302);  mul_302 = None
    clone_40: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_303, memory_format = torch.contiguous_format);  mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_631: "f32[512, 1024]" = torch.ops.aten.view.default(clone_40, [512, 1024]);  clone_40 = None
    permute_381: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    mm_44: "f32[512, 1024]" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
    permute_382: "f32[1024, 512]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_45: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
    permute_383: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[1024]" = torch.ops.aten.view.default(sum_90, [1024]);  sum_90 = None
    permute_384: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_633: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_44, [1, 512, 1024]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_634: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_633, [1, 512, 16, 64]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_635: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_385, [16, 512, 64]);  permute_385 = None
    permute_386: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
    bmm_60: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_635);  permute_386 = None
    permute_387: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None
    bmm_61: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_635, permute_387);  view_635 = permute_387 = None
    view_636: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 16, 512, 64]);  bmm_60 = None
    view_637: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 16, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_12: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_205, torch.float32);  getitem_205 = None
    mul_304: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_305: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_637, mul_304);  view_637 = mul_304 = None
    clone_41: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_305, memory_format = torch.contiguous_format);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_29: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_306: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_41, alias_29);  clone_41 = None
    sum_91: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [-1], True)
    mul_307: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_91);  alias_29 = sum_91 = None
    sub_108: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_108, 8.0);  sub_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_638: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_62, [16, 512, 512]);  div_62 = None
    permute_388: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
    bmm_62: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_638);  permute_388 = None
    permute_389: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None
    bmm_63: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_638, permute_389);  view_638 = permute_389 = None
    view_639: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 16, 64, 512]);  bmm_62 = None
    view_640: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 16, 512, 64]);  bmm_63 = None
    permute_390: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_639, [0, 1, 3, 2]);  view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_641: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_42, [1, 512, 1024]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_642: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_43, [1, 512, 1024]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_643: "f32[512, 1024]" = torch.ops.aten.view.default(view_642, [512, 1024]);  view_642 = None
    permute_393: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    mm_46: "f32[512, 1024]" = torch.ops.aten.mm.default(view_643, permute_393);  permute_393 = None
    permute_394: "f32[1024, 512]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_47: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_394, view_445);  permute_394 = view_445 = None
    permute_395: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
    view_644: "f32[1024]" = torch.ops.aten.view.default(sum_92, [1024]);  sum_92 = None
    permute_396: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_645: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_46, [1, 512, 1024]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_646: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_397, [1, 512, 1024]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_647: "f32[512, 1024]" = torch.ops.aten.view.default(view_646, [512, 1024]);  view_646 = None
    permute_398: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    mm_48: "f32[512, 1024]" = torch.ops.aten.mm.default(view_647, permute_398);  permute_398 = None
    permute_399: "f32[1024, 512]" = torch.ops.aten.permute.default(view_647, [1, 0])
    mm_49: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_399, view_442);  permute_399 = view_442 = None
    permute_400: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_647, [0], True);  view_647 = None
    view_648: "f32[1024]" = torch.ops.aten.view.default(sum_93, [1024]);  sum_93 = None
    permute_401: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_649: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_48, [1, 512, 1024]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_223: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_645, view_649);  view_645 = view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_650: "f32[512, 1024]" = torch.ops.aten.view.default(view_641, [512, 1024]);  view_641 = None
    permute_402: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    mm_50: "f32[512, 1024]" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
    permute_403: "f32[1024, 512]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_51: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
    permute_404: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_94: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[1024]" = torch.ops.aten.view.default(sum_94, [1024]);  sum_94 = None
    permute_405: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_652: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_50, [1, 512, 1024]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_224: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_223, view_652);  add_223 = view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_109: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_161, getitem_203);  add_161 = getitem_203 = None
    mul_308: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_40);  sub_109 = None
    mul_309: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, primals_324);  primals_324 = None
    mul_310: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_309, 1024)
    sum_95: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_309, mul_308);  mul_309 = None
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_308, sum_96);  sum_96 = None
    sub_110: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_310, sum_95);  mul_310 = sum_95 = None
    sub_111: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_110, mul_312);  sub_110 = mul_312 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    mul_313: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_111);  div_63 = sub_111 = None
    mul_314: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_308);  mul_308 = None
    sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_225: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_222, mul_313);  add_222 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
    mul_315: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_316: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_225, mul_315);  mul_315 = None
    clone_44: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_316, memory_format = torch.contiguous_format);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_653: "f32[512, 1024]" = torch.ops.aten.view.default(clone_44, [512, 1024]);  clone_44 = None
    permute_406: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    mm_52: "f32[512, 4096]" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
    permute_407: "f32[1024, 512]" = torch.ops.aten.permute.default(view_653, [1, 0])
    mm_53: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
    permute_408: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
    view_654: "f32[1024]" = torch.ops.aten.view.default(sum_99, [1024]);  sum_99 = None
    permute_409: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_655: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_52, [1, 512, 4096]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_317: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
    erf_30: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
    add_226: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_318: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_226, 0.5);  add_226 = None
    mul_319: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, view_437)
    mul_320: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
    exp_31: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
    mul_321: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_322: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, mul_321);  view_437 = mul_321 = None
    add_227: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
    mul_323: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_655, add_227);  view_655 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_656: "f32[512, 4096]" = torch.ops.aten.view.default(mul_323, [512, 4096]);  mul_323 = None
    permute_410: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    mm_54: "f32[512, 1024]" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
    permute_411: "f32[4096, 512]" = torch.ops.aten.permute.default(view_656, [1, 0])
    mm_55: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
    permute_412: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
    view_657: "f32[4096]" = torch.ops.aten.view.default(sum_100, [4096]);  sum_100 = None
    permute_413: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_658: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_54, [1, 512, 1024]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_112: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_199);  add_157 = getitem_199 = None
    mul_324: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_39);  sub_112 = None
    mul_325: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, primals_318);  primals_318 = None
    mul_326: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, 1024)
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, mul_324);  mul_325 = None
    sum_102: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_324, sum_102);  sum_102 = None
    sub_113: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_326, sum_101);  mul_326 = sum_101 = None
    sub_114: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_113, mul_328);  sub_113 = mul_328 = None
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    mul_329: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_114);  div_64 = sub_114 = None
    mul_330: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, mul_324);  mul_324 = None
    sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_228: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_329);  add_225 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
    mul_331: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_332: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_228, mul_331);  mul_331 = None
    clone_45: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_332, memory_format = torch.contiguous_format);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_659: "f32[512, 1024]" = torch.ops.aten.view.default(clone_45, [512, 1024]);  clone_45 = None
    permute_414: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    mm_56: "f32[512, 1024]" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
    permute_415: "f32[1024, 512]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_57: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
    permute_416: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_105: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[1024]" = torch.ops.aten.view.default(sum_105, [1024]);  sum_105 = None
    permute_417: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_661: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_56, [1, 512, 1024]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_662: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_661, [1, 512, 16, 64]);  view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_663: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_418, [16, 512, 64]);  permute_418 = None
    permute_419: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
    bmm_64: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_663);  permute_419 = None
    permute_420: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    bmm_65: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_663, permute_420);  view_663 = permute_420 = None
    view_664: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 16, 512, 64]);  bmm_64 = None
    view_665: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 16, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_15: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None
    mul_333: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_334: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_665, mul_333);  view_665 = mul_333 = None
    clone_46: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_334, memory_format = torch.contiguous_format);  mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_30: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_335: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_46, alias_30);  clone_46 = None
    sum_106: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [-1], True)
    mul_336: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_30, sum_106);  alias_30 = sum_106 = None
    sub_115: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_65: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_115, 8.0);  sub_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_666: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_65, [16, 512, 512]);  div_65 = None
    permute_421: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
    bmm_66: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_666);  permute_421 = None
    permute_422: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
    bmm_67: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_666, permute_422);  view_666 = permute_422 = None
    view_667: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 16, 64, 512]);  bmm_66 = None
    view_668: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 16, 512, 64]);  bmm_67 = None
    permute_423: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_667, [0, 1, 3, 2]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_669: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_47, [1, 512, 1024]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_670: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_48, [1, 512, 1024]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_671: "f32[512, 1024]" = torch.ops.aten.view.default(view_670, [512, 1024]);  view_670 = None
    permute_426: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    mm_58: "f32[512, 1024]" = torch.ops.aten.mm.default(view_671, permute_426);  permute_426 = None
    permute_427: "f32[1024, 512]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_59: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_427, view_423);  permute_427 = view_423 = None
    permute_428: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[1024]" = torch.ops.aten.view.default(sum_107, [1024]);  sum_107 = None
    permute_429: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_673: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_58, [1, 512, 1024]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_674: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_430, [1, 512, 1024]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_675: "f32[512, 1024]" = torch.ops.aten.view.default(view_674, [512, 1024]);  view_674 = None
    permute_431: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    mm_60: "f32[512, 1024]" = torch.ops.aten.mm.default(view_675, permute_431);  permute_431 = None
    permute_432: "f32[1024, 512]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_61: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_432, view_420);  permute_432 = view_420 = None
    permute_433: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[1024]" = torch.ops.aten.view.default(sum_108, [1024]);  sum_108 = None
    permute_434: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_677: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_60, [1, 512, 1024]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_229: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_673, view_677);  view_673 = view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_678: "f32[512, 1024]" = torch.ops.aten.view.default(view_669, [512, 1024]);  view_669 = None
    permute_435: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    mm_62: "f32[512, 1024]" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
    permute_436: "f32[1024, 512]" = torch.ops.aten.permute.default(view_678, [1, 0])
    mm_63: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
    permute_437: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_109: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
    view_679: "f32[1024]" = torch.ops.aten.view.default(sum_109, [1024]);  sum_109 = None
    permute_438: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_680: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_62, [1, 512, 1024]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_230: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_229, view_680);  add_229 = view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_116: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_193);  add_153 = getitem_193 = None
    mul_337: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_38);  sub_116 = None
    mul_338: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, primals_308);  primals_308 = None
    mul_339: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_338, 1024)
    sum_110: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_338, mul_337);  mul_338 = None
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_337, sum_111);  sum_111 = None
    sub_117: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_339, sum_110);  mul_339 = sum_110 = None
    sub_118: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_117, mul_341);  sub_117 = mul_341 = None
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    mul_342: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_118);  div_66 = sub_118 = None
    mul_343: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, mul_337);  mul_337 = None
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_231: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_228, mul_342);  add_228 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
    mul_344: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_345: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_231, mul_344);  mul_344 = None
    clone_49: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_345, memory_format = torch.contiguous_format);  mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_681: "f32[512, 1024]" = torch.ops.aten.view.default(clone_49, [512, 1024]);  clone_49 = None
    permute_439: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    mm_64: "f32[512, 4096]" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
    permute_440: "f32[1024, 512]" = torch.ops.aten.permute.default(view_681, [1, 0])
    mm_65: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
    permute_441: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_114: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
    view_682: "f32[1024]" = torch.ops.aten.view.default(sum_114, [1024]);  sum_114 = None
    permute_442: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_683: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_64, [1, 512, 4096]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_346: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_31: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_346);  mul_346 = None
    add_232: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_347: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_232, 0.5);  add_232 = None
    mul_348: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_349: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
    exp_32: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_349);  mul_349 = None
    mul_350: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_351: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, mul_350);  view_415 = mul_350 = None
    add_233: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
    mul_352: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_683, add_233);  view_683 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_684: "f32[512, 4096]" = torch.ops.aten.view.default(mul_352, [512, 4096]);  mul_352 = None
    permute_443: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    mm_66: "f32[512, 1024]" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
    permute_444: "f32[4096, 512]" = torch.ops.aten.permute.default(view_684, [1, 0])
    mm_67: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
    permute_445: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
    view_685: "f32[4096]" = torch.ops.aten.view.default(sum_115, [4096]);  sum_115 = None
    permute_446: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_686: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_66, [1, 512, 1024]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_119: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_189);  add_149 = getitem_189 = None
    mul_353: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_37);  sub_119 = None
    mul_354: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, primals_302);  primals_302 = None
    mul_355: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_354, 1024)
    sum_116: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_354, mul_353);  mul_354 = None
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_353, sum_117);  sum_117 = None
    sub_120: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_355, sum_116);  mul_355 = sum_116 = None
    sub_121: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_120, mul_357);  sub_120 = mul_357 = None
    div_67: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    mul_358: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_121);  div_67 = sub_121 = None
    mul_359: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, mul_353);  mul_353 = None
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_686, [0, 1]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_234: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_231, mul_358);  add_231 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None
    mul_360: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_361: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_234, mul_360);  mul_360 = None
    clone_50: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_361, memory_format = torch.contiguous_format);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_687: "f32[512, 1024]" = torch.ops.aten.view.default(clone_50, [512, 1024]);  clone_50 = None
    permute_447: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    mm_68: "f32[512, 1024]" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
    permute_448: "f32[1024, 512]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_69: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
    permute_449: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_120: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
    view_688: "f32[1024]" = torch.ops.aten.view.default(sum_120, [1024]);  sum_120 = None
    permute_450: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_689: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_68, [1, 512, 1024]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_690: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_689, [1, 512, 16, 64]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_691: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_451, [16, 512, 64]);  permute_451 = None
    permute_452: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
    bmm_68: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_691);  permute_452 = None
    permute_453: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    bmm_69: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_691, permute_453);  view_691 = permute_453 = None
    view_692: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 16, 512, 64]);  bmm_68 = None
    view_693: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 16, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_18: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_185, torch.float32);  getitem_185 = None
    mul_362: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_363: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_693, mul_362);  view_693 = mul_362 = None
    clone_51: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_363, memory_format = torch.contiguous_format);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_31: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_364: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_51, alias_31);  clone_51 = None
    sum_121: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [-1], True)
    mul_365: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_121);  alias_31 = sum_121 = None
    sub_122: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_68: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_694: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_68, [16, 512, 512]);  div_68 = None
    permute_454: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
    bmm_70: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_694);  permute_454 = None
    permute_455: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    bmm_71: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_694, permute_455);  view_694 = permute_455 = None
    view_695: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 16, 64, 512]);  bmm_70 = None
    view_696: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 16, 512, 64]);  bmm_71 = None
    permute_456: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_695, [0, 1, 3, 2]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_697: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_52, [1, 512, 1024]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_698: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_53, [1, 512, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_699: "f32[512, 1024]" = torch.ops.aten.view.default(view_698, [512, 1024]);  view_698 = None
    permute_459: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    mm_70: "f32[512, 1024]" = torch.ops.aten.mm.default(view_699, permute_459);  permute_459 = None
    permute_460: "f32[1024, 512]" = torch.ops.aten.permute.default(view_699, [1, 0])
    mm_71: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_460, view_401);  permute_460 = view_401 = None
    permute_461: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_699, [0], True);  view_699 = None
    view_700: "f32[1024]" = torch.ops.aten.view.default(sum_122, [1024]);  sum_122 = None
    permute_462: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_701: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_70, [1, 512, 1024]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_702: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_463, [1, 512, 1024]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_703: "f32[512, 1024]" = torch.ops.aten.view.default(view_702, [512, 1024]);  view_702 = None
    permute_464: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    mm_72: "f32[512, 1024]" = torch.ops.aten.mm.default(view_703, permute_464);  permute_464 = None
    permute_465: "f32[1024, 512]" = torch.ops.aten.permute.default(view_703, [1, 0])
    mm_73: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_465, view_398);  permute_465 = view_398 = None
    permute_466: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_703, [0], True);  view_703 = None
    view_704: "f32[1024]" = torch.ops.aten.view.default(sum_123, [1024]);  sum_123 = None
    permute_467: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_705: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_72, [1, 512, 1024]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_235: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_701, view_705);  view_701 = view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_706: "f32[512, 1024]" = torch.ops.aten.view.default(view_697, [512, 1024]);  view_697 = None
    permute_468: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    mm_74: "f32[512, 1024]" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
    permute_469: "f32[1024, 512]" = torch.ops.aten.permute.default(view_706, [1, 0])
    mm_75: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
    permute_470: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_124: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
    view_707: "f32[1024]" = torch.ops.aten.view.default(sum_124, [1024]);  sum_124 = None
    permute_471: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_708: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_74, [1, 512, 1024]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_236: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_235, view_708);  add_235 = view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_123: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_145, getitem_183);  add_145 = getitem_183 = None
    mul_366: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_36);  sub_123 = None
    mul_367: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, primals_292);  primals_292 = None
    mul_368: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, 1024)
    sum_125: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, mul_366);  mul_367 = None
    sum_126: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_366, sum_126);  sum_126 = None
    sub_124: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_368, sum_125);  mul_368 = sum_125 = None
    sub_125: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_124, mul_370);  sub_124 = mul_370 = None
    div_69: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None
    mul_371: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_125);  div_69 = sub_125 = None
    mul_372: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, mul_366);  mul_366 = None
    sum_127: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_237: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_234, mul_371);  add_234 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None
    mul_373: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_374: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_237, mul_373);  mul_373 = None
    clone_54: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_374, memory_format = torch.contiguous_format);  mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_709: "f32[512, 1024]" = torch.ops.aten.view.default(clone_54, [512, 1024]);  clone_54 = None
    permute_472: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    mm_76: "f32[512, 4096]" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
    permute_473: "f32[1024, 512]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_77: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
    permute_474: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_129: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
    view_710: "f32[1024]" = torch.ops.aten.view.default(sum_129, [1024]);  sum_129 = None
    permute_475: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_711: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_76, [1, 512, 4096]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_375: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_32: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_238: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_376: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_238, 0.5);  add_238 = None
    mul_377: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_378: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
    exp_33: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
    mul_379: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_380: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, mul_379);  view_393 = mul_379 = None
    add_239: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
    mul_381: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_711, add_239);  view_711 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_712: "f32[512, 4096]" = torch.ops.aten.view.default(mul_381, [512, 4096]);  mul_381 = None
    permute_476: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    mm_78: "f32[512, 1024]" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
    permute_477: "f32[4096, 512]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_79: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
    permute_478: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_130: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[4096]" = torch.ops.aten.view.default(sum_130, [4096]);  sum_130 = None
    permute_479: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_714: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_78, [1, 512, 1024]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_126: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_141, getitem_179);  add_141 = getitem_179 = None
    mul_382: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_35);  sub_126 = None
    mul_383: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, primals_286);  primals_286 = None
    mul_384: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, 1024)
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_382);  mul_383 = None
    sum_132: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_382, sum_132);  sum_132 = None
    sub_127: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_384, sum_131);  mul_384 = sum_131 = None
    sub_128: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_127, mul_386);  sub_127 = mul_386 = None
    div_70: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None
    mul_387: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_128);  div_70 = sub_128 = None
    mul_388: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, mul_382);  mul_382 = None
    sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_714, [0, 1]);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_240: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_387);  add_237 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
    mul_389: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_390: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_240, mul_389);  mul_389 = None
    clone_55: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_715: "f32[512, 1024]" = torch.ops.aten.view.default(clone_55, [512, 1024]);  clone_55 = None
    permute_480: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    mm_80: "f32[512, 1024]" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
    permute_481: "f32[1024, 512]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_81: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
    permute_482: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_135: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[1024]" = torch.ops.aten.view.default(sum_135, [1024]);  sum_135 = None
    permute_483: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_717: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_80, [1, 512, 1024]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_718: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_717, [1, 512, 16, 64]);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_719: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_484, [16, 512, 64]);  permute_484 = None
    permute_485: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
    bmm_72: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_719);  permute_485 = None
    permute_486: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
    bmm_73: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_719, permute_486);  view_719 = permute_486 = None
    view_720: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_72, [1, 16, 512, 64]);  bmm_72 = None
    view_721: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_73, [1, 16, 512, 512]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_21: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_175, torch.float32);  getitem_175 = None
    mul_391: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_392: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_721, mul_391);  view_721 = mul_391 = None
    clone_56: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_392, memory_format = torch.contiguous_format);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_32: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_393: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_56, alias_32);  clone_56 = None
    sum_136: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [-1], True)
    mul_394: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_32, sum_136);  alias_32 = sum_136 = None
    sub_129: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_71: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_129, 8.0);  sub_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_722: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_71, [16, 512, 512]);  div_71 = None
    permute_487: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
    bmm_74: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_722);  permute_487 = None
    permute_488: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    bmm_75: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_722, permute_488);  view_722 = permute_488 = None
    view_723: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_74, [1, 16, 64, 512]);  bmm_74 = None
    view_724: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_75, [1, 16, 512, 64]);  bmm_75 = None
    permute_489: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_723, [0, 1, 3, 2]);  view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_725: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_57, [1, 512, 1024]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_726: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_58, [1, 512, 1024]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_727: "f32[512, 1024]" = torch.ops.aten.view.default(view_726, [512, 1024]);  view_726 = None
    permute_492: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    mm_82: "f32[512, 1024]" = torch.ops.aten.mm.default(view_727, permute_492);  permute_492 = None
    permute_493: "f32[1024, 512]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_83: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_493, view_379);  permute_493 = view_379 = None
    permute_494: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_137: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[1024]" = torch.ops.aten.view.default(sum_137, [1024]);  sum_137 = None
    permute_495: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_729: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_82, [1, 512, 1024]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_730: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_496, [1, 512, 1024]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_731: "f32[512, 1024]" = torch.ops.aten.view.default(view_730, [512, 1024]);  view_730 = None
    permute_497: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_84: "f32[512, 1024]" = torch.ops.aten.mm.default(view_731, permute_497);  permute_497 = None
    permute_498: "f32[1024, 512]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_85: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_498, view_376);  permute_498 = view_376 = None
    permute_499: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_731, [0], True);  view_731 = None
    view_732: "f32[1024]" = torch.ops.aten.view.default(sum_138, [1024]);  sum_138 = None
    permute_500: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_733: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_84, [1, 512, 1024]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_241: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_729, view_733);  view_729 = view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_734: "f32[512, 1024]" = torch.ops.aten.view.default(view_725, [512, 1024]);  view_725 = None
    permute_501: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    mm_86: "f32[512, 1024]" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
    permute_502: "f32[1024, 512]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_87: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
    permute_503: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_139: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[1024]" = torch.ops.aten.view.default(sum_139, [1024]);  sum_139 = None
    permute_504: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_736: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_86, [1, 512, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_242: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_241, view_736);  add_241 = view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_130: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_173);  add_137 = getitem_173 = None
    mul_395: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_34);  sub_130 = None
    mul_396: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, primals_276);  primals_276 = None
    mul_397: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_396, 1024)
    sum_140: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_396, mul_395);  mul_396 = None
    sum_141: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_395, sum_141);  sum_141 = None
    sub_131: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_397, sum_140);  mul_397 = sum_140 = None
    sub_132: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_131, mul_399);  sub_131 = mul_399 = None
    div_72: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None
    mul_400: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_132);  div_72 = sub_132 = None
    mul_401: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, mul_395);  mul_395 = None
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_243: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_240, mul_400);  add_240 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
    mul_402: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_403: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_243, mul_402);  mul_402 = None
    clone_59: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_403, memory_format = torch.contiguous_format);  mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_737: "f32[512, 1024]" = torch.ops.aten.view.default(clone_59, [512, 1024]);  clone_59 = None
    permute_505: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm_88: "f32[512, 4096]" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
    permute_506: "f32[1024, 512]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_89: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
    permute_507: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_144: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[1024]" = torch.ops.aten.view.default(sum_144, [1024]);  sum_144 = None
    permute_508: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_739: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_88, [1, 512, 4096]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_404: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_33: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
    add_244: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_405: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_244, 0.5);  add_244 = None
    mul_406: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_407: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_34: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_409: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, mul_408);  view_371 = mul_408 = None
    add_245: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_739, add_245);  view_739 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_740: "f32[512, 4096]" = torch.ops.aten.view.default(mul_410, [512, 4096]);  mul_410 = None
    permute_509: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    mm_90: "f32[512, 1024]" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
    permute_510: "f32[4096, 512]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_91: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
    permute_511: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_145: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[4096]" = torch.ops.aten.view.default(sum_145, [4096]);  sum_145 = None
    permute_512: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_742: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_90, [1, 512, 1024]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_133: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_133, getitem_169);  add_133 = getitem_169 = None
    mul_411: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_33);  sub_133 = None
    mul_412: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, primals_270);  primals_270 = None
    mul_413: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_412, 1024)
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_412, mul_411);  mul_412 = None
    sum_147: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_411, sum_147);  sum_147 = None
    sub_134: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_413, sum_146);  mul_413 = sum_146 = None
    sub_135: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_134, mul_415);  sub_134 = mul_415 = None
    div_73: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None
    mul_416: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_135);  div_73 = sub_135 = None
    mul_417: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, mul_411);  mul_411 = None
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_246: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_243, mul_416);  add_243 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
    mul_418: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_419: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_246, mul_418);  mul_418 = None
    clone_60: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_419, memory_format = torch.contiguous_format);  mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_743: "f32[512, 1024]" = torch.ops.aten.view.default(clone_60, [512, 1024]);  clone_60 = None
    permute_513: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    mm_92: "f32[512, 1024]" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
    permute_514: "f32[1024, 512]" = torch.ops.aten.permute.default(view_743, [1, 0])
    mm_93: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
    permute_515: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
    view_744: "f32[1024]" = torch.ops.aten.view.default(sum_150, [1024]);  sum_150 = None
    permute_516: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_745: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_92, [1, 512, 1024]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_746: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_745, [1, 512, 16, 64]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_747: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_517, [16, 512, 64]);  permute_517 = None
    permute_518: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
    bmm_76: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_747);  permute_518 = None
    permute_519: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    bmm_77: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_747, permute_519);  view_747 = permute_519 = None
    view_748: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_76, [1, 16, 512, 64]);  bmm_76 = None
    view_749: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_77, [1, 16, 512, 512]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_24: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
    mul_420: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_421: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_749, mul_420);  view_749 = mul_420 = None
    clone_61: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_421, memory_format = torch.contiguous_format);  mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_33: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_422: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_61, alias_33);  clone_61 = None
    sum_151: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [-1], True)
    mul_423: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_151);  alias_33 = sum_151 = None
    sub_136: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_74: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_136, 8.0);  sub_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_750: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_74, [16, 512, 512]);  div_74 = None
    permute_520: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
    bmm_78: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_750);  permute_520 = None
    permute_521: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
    bmm_79: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_750, permute_521);  view_750 = permute_521 = None
    view_751: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_78, [1, 16, 64, 512]);  bmm_78 = None
    view_752: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_79, [1, 16, 512, 64]);  bmm_79 = None
    permute_522: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_751, [0, 1, 3, 2]);  view_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_753: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_62, [1, 512, 1024]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_63: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_754: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_63, [1, 512, 1024]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_755: "f32[512, 1024]" = torch.ops.aten.view.default(view_754, [512, 1024]);  view_754 = None
    permute_525: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    mm_94: "f32[512, 1024]" = torch.ops.aten.mm.default(view_755, permute_525);  permute_525 = None
    permute_526: "f32[1024, 512]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_95: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_526, view_357);  permute_526 = view_357 = None
    permute_527: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[1024]" = torch.ops.aten.view.default(sum_152, [1024]);  sum_152 = None
    permute_528: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_757: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_94, [1, 512, 1024]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_758: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_529, [1, 512, 1024]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_759: "f32[512, 1024]" = torch.ops.aten.view.default(view_758, [512, 1024]);  view_758 = None
    permute_530: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    mm_96: "f32[512, 1024]" = torch.ops.aten.mm.default(view_759, permute_530);  permute_530 = None
    permute_531: "f32[1024, 512]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_97: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_531, view_354);  permute_531 = view_354 = None
    permute_532: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
    view_760: "f32[1024]" = torch.ops.aten.view.default(sum_153, [1024]);  sum_153 = None
    permute_533: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_761: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_96, [1, 512, 1024]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_247: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_757, view_761);  view_757 = view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_762: "f32[512, 1024]" = torch.ops.aten.view.default(view_753, [512, 1024]);  view_753 = None
    permute_534: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    mm_98: "f32[512, 1024]" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
    permute_535: "f32[1024, 512]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_99: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
    permute_536: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_154: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
    view_763: "f32[1024]" = torch.ops.aten.view.default(sum_154, [1024]);  sum_154 = None
    permute_537: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_764: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_98, [1, 512, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_248: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_247, view_764);  add_247 = view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_137: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_163);  add_129 = getitem_163 = None
    mul_424: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_32);  sub_137 = None
    mul_425: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, primals_260);  primals_260 = None
    mul_426: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_425, 1024)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_425, mul_424);  mul_425 = None
    sum_156: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_424, sum_156);  sum_156 = None
    sub_138: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_426, sum_155);  mul_426 = sum_155 = None
    sub_139: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_138, mul_428);  sub_138 = mul_428 = None
    div_75: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None
    mul_429: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_139);  div_75 = sub_139 = None
    mul_430: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, mul_424);  mul_424 = None
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_249: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_246, mul_429);  add_246 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
    mul_431: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_432: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_249, mul_431);  mul_431 = None
    clone_64: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_432, memory_format = torch.contiguous_format);  mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_765: "f32[512, 1024]" = torch.ops.aten.view.default(clone_64, [512, 1024]);  clone_64 = None
    permute_538: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    mm_100: "f32[512, 4096]" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
    permute_539: "f32[1024, 512]" = torch.ops.aten.permute.default(view_765, [1, 0])
    mm_101: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
    permute_540: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_159: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
    view_766: "f32[1024]" = torch.ops.aten.view.default(sum_159, [1024]);  sum_159 = None
    permute_541: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    view_767: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_100, [1, 512, 4096]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_433: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
    erf_34: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_433);  mul_433 = None
    add_250: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_434: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_250, 0.5);  add_250 = None
    mul_435: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, view_349)
    mul_436: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
    exp_35: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_436);  mul_436 = None
    mul_437: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_438: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, mul_437);  view_349 = mul_437 = None
    add_251: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
    mul_439: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_767, add_251);  view_767 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_768: "f32[512, 4096]" = torch.ops.aten.view.default(mul_439, [512, 4096]);  mul_439 = None
    permute_542: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    mm_102: "f32[512, 1024]" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
    permute_543: "f32[4096, 512]" = torch.ops.aten.permute.default(view_768, [1, 0])
    mm_103: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
    permute_544: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_160: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
    view_769: "f32[4096]" = torch.ops.aten.view.default(sum_160, [4096]);  sum_160 = None
    permute_545: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_544, [1, 0]);  permute_544 = None
    view_770: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_102, [1, 512, 1024]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_140: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_159);  add_125 = getitem_159 = None
    mul_440: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_31);  sub_140 = None
    mul_441: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, primals_254);  primals_254 = None
    mul_442: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_441, 1024)
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
    mul_443: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_441, mul_440);  mul_441 = None
    sum_162: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
    mul_444: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_440, sum_162);  sum_162 = None
    sub_141: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_442, sum_161);  mul_442 = sum_161 = None
    sub_142: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_141, mul_444);  sub_141 = mul_444 = None
    div_76: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None
    mul_445: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_76, sub_142);  div_76 = sub_142 = None
    mul_446: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, mul_440);  mul_440 = None
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_770, [0, 1]);  view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_252: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_249, mul_445);  add_249 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_447: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_448: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_252, mul_447);  mul_447 = None
    clone_65: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_448, memory_format = torch.contiguous_format);  mul_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_771: "f32[512, 1024]" = torch.ops.aten.view.default(clone_65, [512, 1024]);  clone_65 = None
    permute_546: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_104: "f32[512, 1024]" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
    permute_547: "f32[1024, 512]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_105: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
    permute_548: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_165: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
    view_772: "f32[1024]" = torch.ops.aten.view.default(sum_165, [1024]);  sum_165 = None
    permute_549: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_773: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_104, [1, 512, 1024]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_774: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_773, [1, 512, 16, 64]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_550: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_775: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_550, [16, 512, 64]);  permute_550 = None
    permute_551: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    bmm_80: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_551, view_775);  permute_551 = None
    permute_552: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
    bmm_81: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_775, permute_552);  view_775 = permute_552 = None
    view_776: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_80, [1, 16, 512, 64]);  bmm_80 = None
    view_777: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_81, [1, 16, 512, 512]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_27: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None
    mul_449: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_450: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_777, mul_449);  view_777 = mul_449 = None
    clone_66: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_450, memory_format = torch.contiguous_format);  mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_34: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_451: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_66, alias_34);  clone_66 = None
    sum_166: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
    mul_452: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_34, sum_166);  alias_34 = sum_166 = None
    sub_143: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_77: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_143, 8.0);  sub_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_778: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_77, [16, 512, 512]);  div_77 = None
    permute_553: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
    bmm_82: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_553, view_778);  permute_553 = None
    permute_554: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    bmm_83: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_778, permute_554);  view_778 = permute_554 = None
    view_779: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_82, [1, 16, 64, 512]);  bmm_82 = None
    view_780: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_83, [1, 16, 512, 64]);  bmm_83 = None
    permute_555: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_779, [0, 1, 3, 2]);  view_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_556: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
    view_781: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_67, [1, 512, 1024]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_557: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_68: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    view_782: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_68, [1, 512, 1024]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_783: "f32[512, 1024]" = torch.ops.aten.view.default(view_782, [512, 1024]);  view_782 = None
    permute_558: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_106: "f32[512, 1024]" = torch.ops.aten.mm.default(view_783, permute_558);  permute_558 = None
    permute_559: "f32[1024, 512]" = torch.ops.aten.permute.default(view_783, [1, 0])
    mm_107: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_559, view_335);  permute_559 = view_335 = None
    permute_560: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_167: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_783, [0], True);  view_783 = None
    view_784: "f32[1024]" = torch.ops.aten.view.default(sum_167, [1024]);  sum_167 = None
    permute_561: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_785: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_106, [1, 512, 1024]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_562: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_555, [0, 2, 1, 3]);  permute_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_786: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_562, [1, 512, 1024]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_787: "f32[512, 1024]" = torch.ops.aten.view.default(view_786, [512, 1024]);  view_786 = None
    permute_563: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_108: "f32[512, 1024]" = torch.ops.aten.mm.default(view_787, permute_563);  permute_563 = None
    permute_564: "f32[1024, 512]" = torch.ops.aten.permute.default(view_787, [1, 0])
    mm_109: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_564, view_332);  permute_564 = view_332 = None
    permute_565: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_168: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_787, [0], True);  view_787 = None
    view_788: "f32[1024]" = torch.ops.aten.view.default(sum_168, [1024]);  sum_168 = None
    permute_566: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
    view_789: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_108, [1, 512, 1024]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_253: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_785, view_789);  view_785 = view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_790: "f32[512, 1024]" = torch.ops.aten.view.default(view_781, [512, 1024]);  view_781 = None
    permute_567: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_110: "f32[512, 1024]" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
    permute_568: "f32[1024, 512]" = torch.ops.aten.permute.default(view_790, [1, 0])
    mm_111: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
    permute_569: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_169: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
    view_791: "f32[1024]" = torch.ops.aten.view.default(sum_169, [1024]);  sum_169 = None
    permute_570: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    view_792: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_110, [1, 512, 1024]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_254: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_253, view_792);  add_253 = view_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_144: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_121, getitem_153);  add_121 = getitem_153 = None
    mul_453: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_30);  sub_144 = None
    mul_454: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, primals_244);  primals_244 = None
    mul_455: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_454, 1024)
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_454, mul_453);  mul_454 = None
    sum_171: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_453, sum_171);  sum_171 = None
    sub_145: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_455, sum_170);  mul_455 = sum_170 = None
    sub_146: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_145, mul_457);  sub_145 = mul_457 = None
    div_78: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None
    mul_458: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_78, sub_146);  div_78 = sub_146 = None
    mul_459: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, mul_453);  mul_453 = None
    sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_255: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_252, mul_458);  add_252 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_460: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_461: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_255, mul_460);  mul_460 = None
    clone_69: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_461, memory_format = torch.contiguous_format);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_793: "f32[512, 1024]" = torch.ops.aten.view.default(clone_69, [512, 1024]);  clone_69 = None
    permute_571: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    mm_112: "f32[512, 4096]" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
    permute_572: "f32[1024, 512]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_113: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
    permute_573: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_174: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
    view_794: "f32[1024]" = torch.ops.aten.view.default(sum_174, [1024]);  sum_174 = None
    permute_574: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_795: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_112, [1, 512, 4096]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_462: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_35: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_462);  mul_462 = None
    add_256: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_463: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_256, 0.5);  add_256 = None
    mul_464: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_465: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
    exp_36: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_465);  mul_465 = None
    mul_466: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_467: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, mul_466);  view_327 = mul_466 = None
    add_257: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
    mul_468: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_795, add_257);  view_795 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_796: "f32[512, 4096]" = torch.ops.aten.view.default(mul_468, [512, 4096]);  mul_468 = None
    permute_575: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    mm_114: "f32[512, 1024]" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
    permute_576: "f32[4096, 512]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_115: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
    permute_577: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_175: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[4096]" = torch.ops.aten.view.default(sum_175, [4096]);  sum_175 = None
    permute_578: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_798: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_114, [1, 512, 1024]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_147: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_149);  add_117 = getitem_149 = None
    mul_469: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_29);  sub_147 = None
    mul_470: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, primals_238);  primals_238 = None
    mul_471: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_470, 1024)
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
    mul_472: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_470, mul_469);  mul_470 = None
    sum_177: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
    mul_473: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_469, sum_177);  sum_177 = None
    sub_148: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_471, sum_176);  mul_471 = sum_176 = None
    sub_149: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_148, mul_473);  sub_148 = mul_473 = None
    div_79: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None
    mul_474: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_79, sub_149);  div_79 = sub_149 = None
    mul_475: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, mul_469);  mul_469 = None
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
    sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_798, [0, 1]);  view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_258: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_255, mul_474);  add_255 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_476: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_477: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_258, mul_476);  mul_476 = None
    clone_70: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_477, memory_format = torch.contiguous_format);  mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_799: "f32[512, 1024]" = torch.ops.aten.view.default(clone_70, [512, 1024]);  clone_70 = None
    permute_579: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    mm_116: "f32[512, 1024]" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
    permute_580: "f32[1024, 512]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_117: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
    permute_581: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[1024]" = torch.ops.aten.view.default(sum_180, [1024]);  sum_180 = None
    permute_582: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_801: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_116, [1, 512, 1024]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_802: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_801, [1, 512, 16, 64]);  view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_583: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_803: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_583, [16, 512, 64]);  permute_583 = None
    permute_584: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    bmm_84: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_584, view_803);  permute_584 = None
    permute_585: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    bmm_85: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_803, permute_585);  view_803 = permute_585 = None
    view_804: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_84, [1, 16, 512, 64]);  bmm_84 = None
    view_805: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_85, [1, 16, 512, 512]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_30: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
    mul_478: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_479: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_805, mul_478);  view_805 = mul_478 = None
    clone_71: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_479, memory_format = torch.contiguous_format);  mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_35: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_480: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_71, alias_35);  clone_71 = None
    sum_181: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
    mul_481: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_181);  alias_35 = sum_181 = None
    sub_150: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_80: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_150, 8.0);  sub_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_806: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_80, [16, 512, 512]);  div_80 = None
    permute_586: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
    bmm_86: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_586, view_806);  permute_586 = None
    permute_587: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
    bmm_87: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_806, permute_587);  view_806 = permute_587 = None
    view_807: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_86, [1, 16, 64, 512]);  bmm_86 = None
    view_808: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_87, [1, 16, 512, 64]);  bmm_87 = None
    permute_588: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_807, [0, 1, 3, 2]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_589: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
    view_809: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_72, [1, 512, 1024]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_590: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_73: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_590, memory_format = torch.contiguous_format);  permute_590 = None
    view_810: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_73, [1, 512, 1024]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_811: "f32[512, 1024]" = torch.ops.aten.view.default(view_810, [512, 1024]);  view_810 = None
    permute_591: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    mm_118: "f32[512, 1024]" = torch.ops.aten.mm.default(view_811, permute_591);  permute_591 = None
    permute_592: "f32[1024, 512]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_119: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_592, view_313);  permute_592 = view_313 = None
    permute_593: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[1024]" = torch.ops.aten.view.default(sum_182, [1024]);  sum_182 = None
    permute_594: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_593, [1, 0]);  permute_593 = None
    view_813: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_118, [1, 512, 1024]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_595: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_588, [0, 2, 1, 3]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_814: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_595, [1, 512, 1024]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_815: "f32[512, 1024]" = torch.ops.aten.view.default(view_814, [512, 1024]);  view_814 = None
    permute_596: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    mm_120: "f32[512, 1024]" = torch.ops.aten.mm.default(view_815, permute_596);  permute_596 = None
    permute_597: "f32[1024, 512]" = torch.ops.aten.permute.default(view_815, [1, 0])
    mm_121: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_597, view_310);  permute_597 = view_310 = None
    permute_598: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_815, [0], True);  view_815 = None
    view_816: "f32[1024]" = torch.ops.aten.view.default(sum_183, [1024]);  sum_183 = None
    permute_599: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_817: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_120, [1, 512, 1024]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_259: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_813, view_817);  view_813 = view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_818: "f32[512, 1024]" = torch.ops.aten.view.default(view_809, [512, 1024]);  view_809 = None
    permute_600: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_122: "f32[512, 1024]" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
    permute_601: "f32[1024, 512]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_123: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
    permute_602: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_184: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[1024]" = torch.ops.aten.view.default(sum_184, [1024]);  sum_184 = None
    permute_603: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_820: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_122, [1, 512, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_260: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_259, view_820);  add_259 = view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_151: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_113, getitem_143);  add_113 = getitem_143 = None
    mul_482: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_28);  sub_151 = None
    mul_483: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, primals_228);  primals_228 = None
    mul_484: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_483, 1024)
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_483, mul_482);  mul_483 = None
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_482, sum_186);  sum_186 = None
    sub_152: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_484, sum_185);  mul_484 = sum_185 = None
    sub_153: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_152, mul_486);  sub_152 = mul_486 = None
    div_81: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None
    mul_487: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_81, sub_153);  div_81 = sub_153 = None
    mul_488: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, mul_482);  mul_482 = None
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_261: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_258, mul_487);  add_258 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_489: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_490: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_261, mul_489);  mul_489 = None
    clone_74: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_490, memory_format = torch.contiguous_format);  mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_821: "f32[512, 1024]" = torch.ops.aten.view.default(clone_74, [512, 1024]);  clone_74 = None
    permute_604: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    mm_124: "f32[512, 4096]" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
    permute_605: "f32[1024, 512]" = torch.ops.aten.permute.default(view_821, [1, 0])
    mm_125: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
    permute_606: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_189: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
    view_822: "f32[1024]" = torch.ops.aten.view.default(sum_189, [1024]);  sum_189 = None
    permute_607: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_823: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_124, [1, 512, 4096]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_491: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
    erf_36: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_491);  mul_491 = None
    add_262: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_492: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_262, 0.5);  add_262 = None
    mul_493: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, view_305)
    mul_494: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_493, -0.5);  mul_493 = None
    exp_37: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_494);  mul_494 = None
    mul_495: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_496: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, mul_495);  view_305 = mul_495 = None
    add_263: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None
    mul_497: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_823, add_263);  view_823 = add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_824: "f32[512, 4096]" = torch.ops.aten.view.default(mul_497, [512, 4096]);  mul_497 = None
    permute_608: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_126: "f32[512, 1024]" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
    permute_609: "f32[4096, 512]" = torch.ops.aten.permute.default(view_824, [1, 0])
    mm_127: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
    permute_610: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_190: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
    view_825: "f32[4096]" = torch.ops.aten.view.default(sum_190, [4096]);  sum_190 = None
    permute_611: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_610, [1, 0]);  permute_610 = None
    view_826: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_126, [1, 512, 1024]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_154: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_139);  add_109 = getitem_139 = None
    mul_498: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_27);  sub_154 = None
    mul_499: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, primals_222);  primals_222 = None
    mul_500: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_499, 1024)
    sum_191: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
    mul_501: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_499, mul_498);  mul_499 = None
    sum_192: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
    mul_502: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_498, sum_192);  sum_192 = None
    sub_155: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_500, sum_191);  mul_500 = sum_191 = None
    sub_156: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_155, mul_502);  sub_155 = mul_502 = None
    div_82: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None
    mul_503: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_82, sub_156);  div_82 = sub_156 = None
    mul_504: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, mul_498);  mul_498 = None
    sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_826, [0, 1]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_264: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_261, mul_503);  add_261 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_505: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_506: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_264, mul_505);  mul_505 = None
    clone_75: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_506, memory_format = torch.contiguous_format);  mul_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_827: "f32[512, 1024]" = torch.ops.aten.view.default(clone_75, [512, 1024]);  clone_75 = None
    permute_612: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    mm_128: "f32[512, 1024]" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
    permute_613: "f32[1024, 512]" = torch.ops.aten.permute.default(view_827, [1, 0])
    mm_129: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
    permute_614: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_195: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
    view_828: "f32[1024]" = torch.ops.aten.view.default(sum_195, [1024]);  sum_195 = None
    permute_615: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_614, [1, 0]);  permute_614 = None
    view_829: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_128, [1, 512, 1024]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_830: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_829, [1, 512, 16, 64]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_616: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_831: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_616, [16, 512, 64]);  permute_616 = None
    permute_617: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
    bmm_88: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_617, view_831);  permute_617 = None
    permute_618: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
    bmm_89: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_831, permute_618);  view_831 = permute_618 = None
    view_832: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_88, [1, 16, 512, 64]);  bmm_88 = None
    view_833: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_89, [1, 16, 512, 512]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_33: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
    mul_507: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_508: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_833, mul_507);  view_833 = mul_507 = None
    clone_76: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_508, memory_format = torch.contiguous_format);  mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_36: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_509: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_76, alias_36);  clone_76 = None
    sum_196: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [-1], True)
    mul_510: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_36, sum_196);  alias_36 = sum_196 = None
    sub_157: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_83: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_157, 8.0);  sub_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_834: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_83, [16, 512, 512]);  div_83 = None
    permute_619: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    bmm_90: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_619, view_834);  permute_619 = None
    permute_620: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
    bmm_91: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_834, permute_620);  view_834 = permute_620 = None
    view_835: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_90, [1, 16, 64, 512]);  bmm_90 = None
    view_836: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_91, [1, 16, 512, 64]);  bmm_91 = None
    permute_621: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_835, [0, 1, 3, 2]);  view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_622: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_77: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
    view_837: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_77, [1, 512, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_623: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_78: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_623, memory_format = torch.contiguous_format);  permute_623 = None
    view_838: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_78, [1, 512, 1024]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_839: "f32[512, 1024]" = torch.ops.aten.view.default(view_838, [512, 1024]);  view_838 = None
    permute_624: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_130: "f32[512, 1024]" = torch.ops.aten.mm.default(view_839, permute_624);  permute_624 = None
    permute_625: "f32[1024, 512]" = torch.ops.aten.permute.default(view_839, [1, 0])
    mm_131: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_625, view_291);  permute_625 = view_291 = None
    permute_626: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_839, [0], True);  view_839 = None
    view_840: "f32[1024]" = torch.ops.aten.view.default(sum_197, [1024]);  sum_197 = None
    permute_627: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_626, [1, 0]);  permute_626 = None
    view_841: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_130, [1, 512, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_628: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_621, [0, 2, 1, 3]);  permute_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_842: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_628, [1, 512, 1024]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_843: "f32[512, 1024]" = torch.ops.aten.view.default(view_842, [512, 1024]);  view_842 = None
    permute_629: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_132: "f32[512, 1024]" = torch.ops.aten.mm.default(view_843, permute_629);  permute_629 = None
    permute_630: "f32[1024, 512]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_133: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_630, view_288);  permute_630 = view_288 = None
    permute_631: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[1024]" = torch.ops.aten.view.default(sum_198, [1024]);  sum_198 = None
    permute_632: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_845: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_132, [1, 512, 1024]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_265: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_841, view_845);  view_841 = view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_846: "f32[512, 1024]" = torch.ops.aten.view.default(view_837, [512, 1024]);  view_837 = None
    permute_633: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_134: "f32[512, 1024]" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
    permute_634: "f32[1024, 512]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_135: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
    permute_635: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_199: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
    view_847: "f32[1024]" = torch.ops.aten.view.default(sum_199, [1024]);  sum_199 = None
    permute_636: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
    view_848: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_134, [1, 512, 1024]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_266: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_265, view_848);  add_265 = view_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_158: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_133);  add_105 = getitem_133 = None
    mul_511: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_26);  sub_158 = None
    mul_512: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, primals_212);  primals_212 = None
    mul_513: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_512, 1024)
    sum_200: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True)
    mul_514: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_512, mul_511);  mul_512 = None
    sum_201: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True);  mul_514 = None
    mul_515: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_511, sum_201);  sum_201 = None
    sub_159: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_513, sum_200);  mul_513 = sum_200 = None
    sub_160: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_159, mul_515);  sub_159 = mul_515 = None
    div_84: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
    mul_516: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_84, sub_160);  div_84 = sub_160 = None
    mul_517: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, mul_511);  mul_511 = None
    sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1]);  mul_517 = None
    sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_267: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_264, mul_516);  add_264 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_518: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_519: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_267, mul_518);  mul_518 = None
    clone_79: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_519, memory_format = torch.contiguous_format);  mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_849: "f32[512, 1024]" = torch.ops.aten.view.default(clone_79, [512, 1024]);  clone_79 = None
    permute_637: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_136: "f32[512, 4096]" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
    permute_638: "f32[1024, 512]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_137: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
    permute_639: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_204: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[1024]" = torch.ops.aten.view.default(sum_204, [1024]);  sum_204 = None
    permute_640: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_639, [1, 0]);  permute_639 = None
    view_851: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_136, [1, 512, 4096]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_520: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
    erf_37: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_520);  mul_520 = None
    add_268: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_521: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_268, 0.5);  add_268 = None
    mul_522: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, view_283)
    mul_523: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
    exp_38: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
    mul_524: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_525: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, mul_524);  view_283 = mul_524 = None
    add_269: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
    mul_526: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_851, add_269);  view_851 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_852: "f32[512, 4096]" = torch.ops.aten.view.default(mul_526, [512, 4096]);  mul_526 = None
    permute_641: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_138: "f32[512, 1024]" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
    permute_642: "f32[4096, 512]" = torch.ops.aten.permute.default(view_852, [1, 0])
    mm_139: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
    permute_643: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_205: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
    view_853: "f32[4096]" = torch.ops.aten.view.default(sum_205, [4096]);  sum_205 = None
    permute_644: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_643, [1, 0]);  permute_643 = None
    view_854: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_138, [1, 512, 1024]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_161: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_101, getitem_129);  add_101 = getitem_129 = None
    mul_527: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_25);  sub_161 = None
    mul_528: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, primals_206);  primals_206 = None
    mul_529: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_528, 1024)
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2], True)
    mul_530: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_528, mul_527);  mul_528 = None
    sum_207: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
    mul_531: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_527, sum_207);  sum_207 = None
    sub_162: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_529, sum_206);  mul_529 = sum_206 = None
    sub_163: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_162, mul_531);  sub_162 = mul_531 = None
    div_85: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
    mul_532: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_85, sub_163);  div_85 = sub_163 = None
    mul_533: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, mul_527);  mul_527 = None
    sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 1]);  mul_533 = None
    sum_209: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_270: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_267, mul_532);  add_267 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_534: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_535: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_270, mul_534);  mul_534 = None
    clone_80: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_535, memory_format = torch.contiguous_format);  mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_855: "f32[512, 1024]" = torch.ops.aten.view.default(clone_80, [512, 1024]);  clone_80 = None
    permute_645: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_140: "f32[512, 1024]" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
    permute_646: "f32[1024, 512]" = torch.ops.aten.permute.default(view_855, [1, 0])
    mm_141: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
    permute_647: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_210: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
    view_856: "f32[1024]" = torch.ops.aten.view.default(sum_210, [1024]);  sum_210 = None
    permute_648: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_647, [1, 0]);  permute_647 = None
    view_857: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_140, [1, 512, 1024]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_858: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_857, [1, 512, 16, 64]);  view_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_649: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_859: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_649, [16, 512, 64]);  permute_649 = None
    permute_650: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
    bmm_92: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_650, view_859);  permute_650 = None
    permute_651: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    bmm_93: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_859, permute_651);  view_859 = permute_651 = None
    view_860: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_92, [1, 16, 512, 64]);  bmm_92 = None
    view_861: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_93, [1, 16, 512, 512]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_36: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_536: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_537: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_861, mul_536);  view_861 = mul_536 = None
    clone_81: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_537, memory_format = torch.contiguous_format);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_37: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_538: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_81, alias_37);  clone_81 = None
    sum_211: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [-1], True)
    mul_539: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_37, sum_211);  alias_37 = sum_211 = None
    sub_164: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_86: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_164, 8.0);  sub_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_862: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_86, [16, 512, 512]);  div_86 = None
    permute_652: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
    bmm_94: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_652, view_862);  permute_652 = None
    permute_653: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
    bmm_95: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_862, permute_653);  view_862 = permute_653 = None
    view_863: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_94, [1, 16, 64, 512]);  bmm_94 = None
    view_864: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_95, [1, 16, 512, 64]);  bmm_95 = None
    permute_654: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_863, [0, 1, 3, 2]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_655: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_864, [0, 2, 1, 3]);  view_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_82: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
    view_865: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_82, [1, 512, 1024]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_656: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_83: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
    view_866: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_83, [1, 512, 1024]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_867: "f32[512, 1024]" = torch.ops.aten.view.default(view_866, [512, 1024]);  view_866 = None
    permute_657: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_142: "f32[512, 1024]" = torch.ops.aten.mm.default(view_867, permute_657);  permute_657 = None
    permute_658: "f32[1024, 512]" = torch.ops.aten.permute.default(view_867, [1, 0])
    mm_143: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_658, view_269);  permute_658 = view_269 = None
    permute_659: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_867, [0], True);  view_867 = None
    view_868: "f32[1024]" = torch.ops.aten.view.default(sum_212, [1024]);  sum_212 = None
    permute_660: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_659, [1, 0]);  permute_659 = None
    view_869: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_142, [1, 512, 1024]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_661: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_654, [0, 2, 1, 3]);  permute_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_870: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_661, [1, 512, 1024]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_871: "f32[512, 1024]" = torch.ops.aten.view.default(view_870, [512, 1024]);  view_870 = None
    permute_662: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_144: "f32[512, 1024]" = torch.ops.aten.mm.default(view_871, permute_662);  permute_662 = None
    permute_663: "f32[1024, 512]" = torch.ops.aten.permute.default(view_871, [1, 0])
    mm_145: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_663, view_266);  permute_663 = view_266 = None
    permute_664: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True);  view_871 = None
    view_872: "f32[1024]" = torch.ops.aten.view.default(sum_213, [1024]);  sum_213 = None
    permute_665: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_664, [1, 0]);  permute_664 = None
    view_873: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_144, [1, 512, 1024]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_271: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_869, view_873);  view_869 = view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_874: "f32[512, 1024]" = torch.ops.aten.view.default(view_865, [512, 1024]);  view_865 = None
    permute_666: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_146: "f32[512, 1024]" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
    permute_667: "f32[1024, 512]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_147: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
    permute_668: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_214: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[1024]" = torch.ops.aten.view.default(sum_214, [1024]);  sum_214 = None
    permute_669: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_668, [1, 0]);  permute_668 = None
    view_876: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_146, [1, 512, 1024]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_272: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_271, view_876);  add_271 = view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_165: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  add_97 = getitem_123 = None
    mul_540: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_24);  sub_165 = None
    mul_541: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, primals_196);  primals_196 = None
    mul_542: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_541, 1024)
    sum_215: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_541, [2], True)
    mul_543: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_541, mul_540);  mul_541 = None
    sum_216: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True);  mul_543 = None
    mul_544: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_540, sum_216);  sum_216 = None
    sub_166: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_542, sum_215);  mul_542 = sum_215 = None
    sub_167: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_166, mul_544);  sub_166 = mul_544 = None
    div_87: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
    mul_545: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_87, sub_167);  div_87 = sub_167 = None
    mul_546: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, mul_540);  mul_540 = None
    sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 1]);  mul_546 = None
    sum_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_273: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_270, mul_545);  add_270 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_37: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_547: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_548: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_273, mul_547);  mul_547 = None
    clone_84: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_548, memory_format = torch.contiguous_format);  mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_877: "f32[512, 1024]" = torch.ops.aten.view.default(clone_84, [512, 1024]);  clone_84 = None
    permute_670: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_148: "f32[512, 4096]" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
    permute_671: "f32[1024, 512]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_149: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
    permute_672: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_219: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[1024]" = torch.ops.aten.view.default(sum_219, [1024]);  sum_219 = None
    permute_673: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
    view_879: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_148, [1, 512, 4096]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_549: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_38: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_549);  mul_549 = None
    add_274: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_550: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_274, 0.5);  add_274 = None
    mul_551: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_552: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_551, -0.5);  mul_551 = None
    exp_39: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_552);  mul_552 = None
    mul_553: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_554: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, mul_553);  view_261 = mul_553 = None
    add_275: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_550, mul_554);  mul_550 = mul_554 = None
    mul_555: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_879, add_275);  view_879 = add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_880: "f32[512, 4096]" = torch.ops.aten.view.default(mul_555, [512, 4096]);  mul_555 = None
    permute_674: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_150: "f32[512, 1024]" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
    permute_675: "f32[4096, 512]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_151: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
    permute_676: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_220: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[4096]" = torch.ops.aten.view.default(sum_220, [4096]);  sum_220 = None
    permute_677: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_676, [1, 0]);  permute_676 = None
    view_882: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_150, [1, 512, 1024]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_168: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  add_93 = getitem_119 = None
    mul_556: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_23);  sub_168 = None
    mul_557: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, primals_190);  primals_190 = None
    mul_558: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_557, 1024)
    sum_221: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
    mul_559: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_557, mul_556);  mul_557 = None
    sum_222: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
    mul_560: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_556, sum_222);  sum_222 = None
    sub_169: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_558, sum_221);  mul_558 = sum_221 = None
    sub_170: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_169, mul_560);  sub_169 = mul_560 = None
    div_88: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
    mul_561: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_88, sub_170);  div_88 = sub_170 = None
    mul_562: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, mul_556);  mul_556 = None
    sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
    sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_276: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_273, mul_561);  add_273 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_563: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_564: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_276, mul_563);  mul_563 = None
    clone_85: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_564, memory_format = torch.contiguous_format);  mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_883: "f32[512, 1024]" = torch.ops.aten.view.default(clone_85, [512, 1024]);  clone_85 = None
    permute_678: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_152: "f32[512, 1024]" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
    permute_679: "f32[1024, 512]" = torch.ops.aten.permute.default(view_883, [1, 0])
    mm_153: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
    permute_680: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_225: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
    view_884: "f32[1024]" = torch.ops.aten.view.default(sum_225, [1024]);  sum_225 = None
    permute_681: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_680, [1, 0]);  permute_680 = None
    view_885: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_152, [1, 512, 1024]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_886: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_885, [1, 512, 16, 64]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_682: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_887: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_682, [16, 512, 64]);  permute_682 = None
    permute_683: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_96: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_683, view_887);  permute_683 = None
    permute_684: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    bmm_97: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_887, permute_684);  view_887 = permute_684 = None
    view_888: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_96, [1, 16, 512, 64]);  bmm_96 = None
    view_889: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_97, [1, 16, 512, 512]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_39: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_565: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_566: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_889, mul_565);  view_889 = mul_565 = None
    clone_86: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_566, memory_format = torch.contiguous_format);  mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_38: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_567: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_86, alias_38);  clone_86 = None
    sum_226: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [-1], True)
    mul_568: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_38, sum_226);  alias_38 = sum_226 = None
    sub_171: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_89: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_171, 8.0);  sub_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_890: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_89, [16, 512, 512]);  div_89 = None
    permute_685: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_98: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_685, view_890);  permute_685 = None
    permute_686: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_99: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_890, permute_686);  view_890 = permute_686 = None
    view_891: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_98, [1, 16, 64, 512]);  bmm_98 = None
    view_892: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_99, [1, 16, 512, 64]);  bmm_99 = None
    permute_687: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_891, [0, 1, 3, 2]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_688: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_87: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
    view_893: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_87, [1, 512, 1024]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_689: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_88: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
    view_894: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_88, [1, 512, 1024]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_895: "f32[512, 1024]" = torch.ops.aten.view.default(view_894, [512, 1024]);  view_894 = None
    permute_690: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_154: "f32[512, 1024]" = torch.ops.aten.mm.default(view_895, permute_690);  permute_690 = None
    permute_691: "f32[1024, 512]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_155: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_691, view_247);  permute_691 = view_247 = None
    permute_692: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_227: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[1024]" = torch.ops.aten.view.default(sum_227, [1024]);  sum_227 = None
    permute_693: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    view_897: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_154, [1, 512, 1024]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_694: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_687, [0, 2, 1, 3]);  permute_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_898: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_694, [1, 512, 1024]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_899: "f32[512, 1024]" = torch.ops.aten.view.default(view_898, [512, 1024]);  view_898 = None
    permute_695: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_156: "f32[512, 1024]" = torch.ops.aten.mm.default(view_899, permute_695);  permute_695 = None
    permute_696: "f32[1024, 512]" = torch.ops.aten.permute.default(view_899, [1, 0])
    mm_157: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_696, view_244);  permute_696 = view_244 = None
    permute_697: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_899, [0], True);  view_899 = None
    view_900: "f32[1024]" = torch.ops.aten.view.default(sum_228, [1024]);  sum_228 = None
    permute_698: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_697, [1, 0]);  permute_697 = None
    view_901: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_156, [1, 512, 1024]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_277: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_897, view_901);  view_897 = view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_902: "f32[512, 1024]" = torch.ops.aten.view.default(view_893, [512, 1024]);  view_893 = None
    permute_699: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_158: "f32[512, 1024]" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
    permute_700: "f32[1024, 512]" = torch.ops.aten.permute.default(view_902, [1, 0])
    mm_159: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
    permute_701: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_229: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
    view_903: "f32[1024]" = torch.ops.aten.view.default(sum_229, [1024]);  sum_229 = None
    permute_702: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_701, [1, 0]);  permute_701 = None
    view_904: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_158, [1, 512, 1024]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_278: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_277, view_904);  add_277 = view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_172: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  add_89 = getitem_113 = None
    mul_569: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_22);  sub_172 = None
    mul_570: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, primals_180);  primals_180 = None
    mul_571: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_570, 1024)
    sum_230: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True)
    mul_572: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_570, mul_569);  mul_570 = None
    sum_231: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
    mul_573: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_569, sum_231);  sum_231 = None
    sub_173: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_571, sum_230);  mul_571 = sum_230 = None
    sub_174: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_173, mul_573);  sub_173 = mul_573 = None
    div_90: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
    mul_574: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_90, sub_174);  div_90 = sub_174 = None
    mul_575: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_569);  mul_569 = None
    sum_232: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
    sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_279: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_276, mul_574);  add_276 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_576: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_577: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_279, mul_576);  mul_576 = None
    clone_89: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_577, memory_format = torch.contiguous_format);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_905: "f32[512, 1024]" = torch.ops.aten.view.default(clone_89, [512, 1024]);  clone_89 = None
    permute_703: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_160: "f32[512, 4096]" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
    permute_704: "f32[1024, 512]" = torch.ops.aten.permute.default(view_905, [1, 0])
    mm_161: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
    permute_705: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_234: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
    view_906: "f32[1024]" = torch.ops.aten.view.default(sum_234, [1024]);  sum_234 = None
    permute_706: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
    view_907: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_160, [1, 512, 4096]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_578: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_39: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_578);  mul_578 = None
    add_280: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_579: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_280, 0.5);  add_280 = None
    mul_580: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_581: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_580, -0.5);  mul_580 = None
    exp_40: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_581);  mul_581 = None
    mul_582: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_583: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, mul_582);  view_239 = mul_582 = None
    add_281: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_579, mul_583);  mul_579 = mul_583 = None
    mul_584: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_907, add_281);  view_907 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_908: "f32[512, 4096]" = torch.ops.aten.view.default(mul_584, [512, 4096]);  mul_584 = None
    permute_707: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_162: "f32[512, 1024]" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
    permute_708: "f32[4096, 512]" = torch.ops.aten.permute.default(view_908, [1, 0])
    mm_163: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
    permute_709: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_235: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
    view_909: "f32[4096]" = torch.ops.aten.view.default(sum_235, [4096]);  sum_235 = None
    permute_710: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_709, [1, 0]);  permute_709 = None
    view_910: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_162, [1, 512, 1024]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_175: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  add_85 = getitem_109 = None
    mul_585: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_175, rsqrt_21);  sub_175 = None
    mul_586: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, primals_174);  primals_174 = None
    mul_587: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_586, 1024)
    sum_236: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True)
    mul_588: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_586, mul_585);  mul_586 = None
    sum_237: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True);  mul_588 = None
    mul_589: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_585, sum_237);  sum_237 = None
    sub_176: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_587, sum_236);  mul_587 = sum_236 = None
    sub_177: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_176, mul_589);  sub_176 = mul_589 = None
    div_91: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
    mul_590: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_91, sub_177);  div_91 = sub_177 = None
    mul_591: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, mul_585);  mul_585 = None
    sum_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1]);  mul_591 = None
    sum_239: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_910, [0, 1]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_282: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_279, mul_590);  add_279 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_41: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_592: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_593: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_282, mul_592);  mul_592 = None
    clone_90: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_593, memory_format = torch.contiguous_format);  mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_911: "f32[512, 1024]" = torch.ops.aten.view.default(clone_90, [512, 1024]);  clone_90 = None
    permute_711: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_164: "f32[512, 1024]" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
    permute_712: "f32[1024, 512]" = torch.ops.aten.permute.default(view_911, [1, 0])
    mm_165: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
    permute_713: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
    view_912: "f32[1024]" = torch.ops.aten.view.default(sum_240, [1024]);  sum_240 = None
    permute_714: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_713, [1, 0]);  permute_713 = None
    view_913: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_164, [1, 512, 1024]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_914: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_913, [1, 512, 16, 64]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_715: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_915: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_715, [16, 512, 64]);  permute_715 = None
    permute_716: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_100: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_716, view_915);  permute_716 = None
    permute_717: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_101: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_915, permute_717);  view_915 = permute_717 = None
    view_916: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_100, [1, 16, 512, 64]);  bmm_100 = None
    view_917: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_101, [1, 16, 512, 512]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_42: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_594: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_595: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_917, mul_594);  view_917 = mul_594 = None
    clone_91: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_595, memory_format = torch.contiguous_format);  mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_39: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_596: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_91, alias_39);  clone_91 = None
    sum_241: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [-1], True)
    mul_597: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_241);  alias_39 = sum_241 = None
    sub_178: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_92: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_178, 8.0);  sub_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_918: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_92, [16, 512, 512]);  div_92 = None
    permute_718: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    bmm_102: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_718, view_918);  permute_718 = None
    permute_719: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    bmm_103: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_918, permute_719);  view_918 = permute_719 = None
    view_919: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_102, [1, 16, 64, 512]);  bmm_102 = None
    view_920: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_103, [1, 16, 512, 64]);  bmm_103 = None
    permute_720: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_919, [0, 1, 3, 2]);  view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_721: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_920, [0, 2, 1, 3]);  view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_92: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
    view_921: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_92, [1, 512, 1024]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_722: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_916, [0, 2, 1, 3]);  view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_93: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_722, memory_format = torch.contiguous_format);  permute_722 = None
    view_922: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_93, [1, 512, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_923: "f32[512, 1024]" = torch.ops.aten.view.default(view_922, [512, 1024]);  view_922 = None
    permute_723: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_166: "f32[512, 1024]" = torch.ops.aten.mm.default(view_923, permute_723);  permute_723 = None
    permute_724: "f32[1024, 512]" = torch.ops.aten.permute.default(view_923, [1, 0])
    mm_167: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_724, view_225);  permute_724 = view_225 = None
    permute_725: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_923, [0], True);  view_923 = None
    view_924: "f32[1024]" = torch.ops.aten.view.default(sum_242, [1024]);  sum_242 = None
    permute_726: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_725, [1, 0]);  permute_725 = None
    view_925: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_166, [1, 512, 1024]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_727: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_720, [0, 2, 1, 3]);  permute_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_926: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_727, [1, 512, 1024]);  permute_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_927: "f32[512, 1024]" = torch.ops.aten.view.default(view_926, [512, 1024]);  view_926 = None
    permute_728: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_168: "f32[512, 1024]" = torch.ops.aten.mm.default(view_927, permute_728);  permute_728 = None
    permute_729: "f32[1024, 512]" = torch.ops.aten.permute.default(view_927, [1, 0])
    mm_169: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_729, view_222);  permute_729 = view_222 = None
    permute_730: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_927, [0], True);  view_927 = None
    view_928: "f32[1024]" = torch.ops.aten.view.default(sum_243, [1024]);  sum_243 = None
    permute_731: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_730, [1, 0]);  permute_730 = None
    view_929: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_168, [1, 512, 1024]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_283: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_925, view_929);  view_925 = view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_930: "f32[512, 1024]" = torch.ops.aten.view.default(view_921, [512, 1024]);  view_921 = None
    permute_732: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_170: "f32[512, 1024]" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
    permute_733: "f32[1024, 512]" = torch.ops.aten.permute.default(view_930, [1, 0])
    mm_171: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
    permute_734: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_244: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
    view_931: "f32[1024]" = torch.ops.aten.view.default(sum_244, [1024]);  sum_244 = None
    permute_735: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_932: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_170, [1, 512, 1024]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_284: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_283, view_932);  add_283 = view_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_179: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  add_81 = getitem_103 = None
    mul_598: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_179, rsqrt_20);  sub_179 = None
    mul_599: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, primals_164);  primals_164 = None
    mul_600: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_599, 1024)
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True)
    mul_601: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_599, mul_598);  mul_599 = None
    sum_246: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True);  mul_601 = None
    mul_602: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_598, sum_246);  sum_246 = None
    sub_180: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_600, sum_245);  mul_600 = sum_245 = None
    sub_181: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_180, mul_602);  sub_180 = mul_602 = None
    div_93: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
    mul_603: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_93, sub_181);  div_93 = sub_181 = None
    mul_604: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, mul_598);  mul_598 = None
    sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 1]);  mul_604 = None
    sum_248: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_285: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_282, mul_603);  add_282 = mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_43: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_605: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_606: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_285, mul_605);  mul_605 = None
    clone_94: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_606, memory_format = torch.contiguous_format);  mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_933: "f32[512, 1024]" = torch.ops.aten.view.default(clone_94, [512, 1024]);  clone_94 = None
    permute_736: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_172: "f32[512, 4096]" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
    permute_737: "f32[1024, 512]" = torch.ops.aten.permute.default(view_933, [1, 0])
    mm_173: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
    permute_738: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_249: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
    view_934: "f32[1024]" = torch.ops.aten.view.default(sum_249, [1024]);  sum_249 = None
    permute_739: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_935: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_172, [1, 512, 4096]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_607: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_40: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_607);  mul_607 = None
    add_286: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_608: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_286, 0.5);  add_286 = None
    mul_609: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_610: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_609, -0.5);  mul_609 = None
    exp_41: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_610);  mul_610 = None
    mul_611: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_612: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, mul_611);  view_217 = mul_611 = None
    add_287: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_608, mul_612);  mul_608 = mul_612 = None
    mul_613: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_935, add_287);  view_935 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_936: "f32[512, 4096]" = torch.ops.aten.view.default(mul_613, [512, 4096]);  mul_613 = None
    permute_740: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_174: "f32[512, 1024]" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
    permute_741: "f32[4096, 512]" = torch.ops.aten.permute.default(view_936, [1, 0])
    mm_175: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
    permute_742: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_250: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
    view_937: "f32[4096]" = torch.ops.aten.view.default(sum_250, [4096]);  sum_250 = None
    permute_743: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    view_938: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_174, [1, 512, 1024]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_182: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  add_77 = getitem_99 = None
    mul_614: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_182, rsqrt_19);  sub_182 = None
    mul_615: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, primals_158);  primals_158 = None
    mul_616: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_615, 1024)
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True)
    mul_617: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_615, mul_614);  mul_615 = None
    sum_252: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True);  mul_617 = None
    mul_618: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_614, sum_252);  sum_252 = None
    sub_183: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_616, sum_251);  mul_616 = sum_251 = None
    sub_184: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_183, mul_618);  sub_183 = mul_618 = None
    div_94: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
    mul_619: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_94, sub_184);  div_94 = sub_184 = None
    mul_620: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, mul_614);  mul_614 = None
    sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1]);  mul_620 = None
    sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_938, [0, 1]);  view_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_288: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_285, mul_619);  add_285 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_44: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_621: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_622: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_288, mul_621);  mul_621 = None
    clone_95: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_622, memory_format = torch.contiguous_format);  mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_939: "f32[512, 1024]" = torch.ops.aten.view.default(clone_95, [512, 1024]);  clone_95 = None
    permute_744: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_176: "f32[512, 1024]" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
    permute_745: "f32[1024, 512]" = torch.ops.aten.permute.default(view_939, [1, 0])
    mm_177: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
    permute_746: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
    view_940: "f32[1024]" = torch.ops.aten.view.default(sum_255, [1024]);  sum_255 = None
    permute_747: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_746, [1, 0]);  permute_746 = None
    view_941: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_176, [1, 512, 1024]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_942: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_941, [1, 512, 16, 64]);  view_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_748: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_943: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_748, [16, 512, 64]);  permute_748 = None
    permute_749: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_104: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_749, view_943);  permute_749 = None
    permute_750: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    bmm_105: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_943, permute_750);  view_943 = permute_750 = None
    view_944: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_104, [1, 16, 512, 64]);  bmm_104 = None
    view_945: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_105, [1, 16, 512, 512]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_45: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_623: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_624: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_945, mul_623);  view_945 = mul_623 = None
    clone_96: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_624, memory_format = torch.contiguous_format);  mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_40: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_625: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_96, alias_40);  clone_96 = None
    sum_256: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_625, [-1], True)
    mul_626: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_40, sum_256);  alias_40 = sum_256 = None
    sub_185: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_625, mul_626);  mul_625 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_95: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_185, 8.0);  sub_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_946: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_95, [16, 512, 512]);  div_95 = None
    permute_751: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_106: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_751, view_946);  permute_751 = None
    permute_752: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_107: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_946, permute_752);  view_946 = permute_752 = None
    view_947: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_106, [1, 16, 64, 512]);  bmm_106 = None
    view_948: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_107, [1, 16, 512, 64]);  bmm_107 = None
    permute_753: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_947, [0, 1, 3, 2]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_754: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_97: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_949: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_97, [1, 512, 1024]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_755: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_944, [0, 2, 1, 3]);  view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_98: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
    view_950: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_98, [1, 512, 1024]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_951: "f32[512, 1024]" = torch.ops.aten.view.default(view_950, [512, 1024]);  view_950 = None
    permute_756: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_178: "f32[512, 1024]" = torch.ops.aten.mm.default(view_951, permute_756);  permute_756 = None
    permute_757: "f32[1024, 512]" = torch.ops.aten.permute.default(view_951, [1, 0])
    mm_179: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_757, view_203);  permute_757 = view_203 = None
    permute_758: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_257: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_951, [0], True);  view_951 = None
    view_952: "f32[1024]" = torch.ops.aten.view.default(sum_257, [1024]);  sum_257 = None
    permute_759: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    view_953: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_178, [1, 512, 1024]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_760: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_753, [0, 2, 1, 3]);  permute_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_954: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_760, [1, 512, 1024]);  permute_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_955: "f32[512, 1024]" = torch.ops.aten.view.default(view_954, [512, 1024]);  view_954 = None
    permute_761: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_180: "f32[512, 1024]" = torch.ops.aten.mm.default(view_955, permute_761);  permute_761 = None
    permute_762: "f32[1024, 512]" = torch.ops.aten.permute.default(view_955, [1, 0])
    mm_181: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_762, view_200);  permute_762 = view_200 = None
    permute_763: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_955, [0], True);  view_955 = None
    view_956: "f32[1024]" = torch.ops.aten.view.default(sum_258, [1024]);  sum_258 = None
    permute_764: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_763, [1, 0]);  permute_763 = None
    view_957: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_180, [1, 512, 1024]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_289: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_953, view_957);  view_953 = view_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_958: "f32[512, 1024]" = torch.ops.aten.view.default(view_949, [512, 1024]);  view_949 = None
    permute_765: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_182: "f32[512, 1024]" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
    permute_766: "f32[1024, 512]" = torch.ops.aten.permute.default(view_958, [1, 0])
    mm_183: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
    permute_767: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_259: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
    view_959: "f32[1024]" = torch.ops.aten.view.default(sum_259, [1024]);  sum_259 = None
    permute_768: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_767, [1, 0]);  permute_767 = None
    view_960: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_182, [1, 512, 1024]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_290: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_289, view_960);  add_289 = view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_186: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  add_73 = getitem_93 = None
    mul_627: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_186, rsqrt_18);  sub_186 = None
    mul_628: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_148);  primals_148 = None
    mul_629: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_628, 1024)
    sum_260: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True)
    mul_630: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_628, mul_627);  mul_628 = None
    sum_261: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True);  mul_630 = None
    mul_631: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_627, sum_261);  sum_261 = None
    sub_187: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_629, sum_260);  mul_629 = sum_260 = None
    sub_188: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_187, mul_631);  sub_187 = mul_631 = None
    div_96: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
    mul_632: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_96, sub_188);  div_96 = sub_188 = None
    mul_633: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_627);  mul_627 = None
    sum_262: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 1]);  mul_633 = None
    sum_263: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_291: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_288, mul_632);  add_288 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_634: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_635: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_291, mul_634);  mul_634 = None
    clone_99: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_635, memory_format = torch.contiguous_format);  mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_961: "f32[512, 1024]" = torch.ops.aten.view.default(clone_99, [512, 1024]);  clone_99 = None
    permute_769: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_184: "f32[512, 4096]" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
    permute_770: "f32[1024, 512]" = torch.ops.aten.permute.default(view_961, [1, 0])
    mm_185: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
    permute_771: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_264: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
    view_962: "f32[1024]" = torch.ops.aten.view.default(sum_264, [1024]);  sum_264 = None
    permute_772: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_771, [1, 0]);  permute_771 = None
    view_963: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_184, [1, 512, 4096]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_636: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_41: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_636);  mul_636 = None
    add_292: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_637: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_292, 0.5);  add_292 = None
    mul_638: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_639: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_638, -0.5);  mul_638 = None
    exp_42: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_639);  mul_639 = None
    mul_640: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_641: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, mul_640);  view_195 = mul_640 = None
    add_293: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_637, mul_641);  mul_637 = mul_641 = None
    mul_642: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_963, add_293);  view_963 = add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_964: "f32[512, 4096]" = torch.ops.aten.view.default(mul_642, [512, 4096]);  mul_642 = None
    permute_773: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_186: "f32[512, 1024]" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
    permute_774: "f32[4096, 512]" = torch.ops.aten.permute.default(view_964, [1, 0])
    mm_187: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
    permute_775: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_265: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
    view_965: "f32[4096]" = torch.ops.aten.view.default(sum_265, [4096]);  sum_265 = None
    permute_776: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_775, [1, 0]);  permute_775 = None
    view_966: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_186, [1, 512, 1024]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_189: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  add_69 = getitem_89 = None
    mul_643: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_189, rsqrt_17);  sub_189 = None
    mul_644: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, primals_142);  primals_142 = None
    mul_645: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_644, 1024)
    sum_266: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
    mul_646: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_644, mul_643);  mul_644 = None
    sum_267: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
    mul_647: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_643, sum_267);  sum_267 = None
    sub_190: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_645, sum_266);  mul_645 = sum_266 = None
    sub_191: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_190, mul_647);  sub_190 = mul_647 = None
    div_97: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None
    mul_648: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_97, sub_191);  div_97 = sub_191 = None
    mul_649: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, mul_643);  mul_643 = None
    sum_268: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
    sum_269: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_294: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_291, mul_648);  add_291 = mul_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_47: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_650: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_651: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_294, mul_650);  mul_650 = None
    clone_100: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_651, memory_format = torch.contiguous_format);  mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_967: "f32[512, 1024]" = torch.ops.aten.view.default(clone_100, [512, 1024]);  clone_100 = None
    permute_777: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_188: "f32[512, 1024]" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
    permute_778: "f32[1024, 512]" = torch.ops.aten.permute.default(view_967, [1, 0])
    mm_189: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
    permute_779: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_270: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
    view_968: "f32[1024]" = torch.ops.aten.view.default(sum_270, [1024]);  sum_270 = None
    permute_780: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
    view_969: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_188, [1, 512, 1024]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_970: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_969, [1, 512, 16, 64]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_781: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_971: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_781, [16, 512, 64]);  permute_781 = None
    permute_782: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_108: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_782, view_971);  permute_782 = None
    permute_783: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_109: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_971, permute_783);  view_971 = permute_783 = None
    view_972: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_108, [1, 16, 512, 64]);  bmm_108 = None
    view_973: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_109, [1, 16, 512, 512]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_48: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_652: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_653: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_973, mul_652);  view_973 = mul_652 = None
    clone_101: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_653, memory_format = torch.contiguous_format);  mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_41: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_654: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_101, alias_41);  clone_101 = None
    sum_271: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_654, [-1], True)
    mul_655: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_41, sum_271);  alias_41 = sum_271 = None
    sub_192: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_98: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_192, 8.0);  sub_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_974: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_98, [16, 512, 512]);  div_98 = None
    permute_784: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_110: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_784, view_974);  permute_784 = None
    permute_785: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_111: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_974, permute_785);  view_974 = permute_785 = None
    view_975: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_110, [1, 16, 64, 512]);  bmm_110 = None
    view_976: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_111, [1, 16, 512, 64]);  bmm_111 = None
    permute_786: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_975, [0, 1, 3, 2]);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_787: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_102: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
    view_977: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_102, [1, 512, 1024]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_788: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_103: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_788, memory_format = torch.contiguous_format);  permute_788 = None
    view_978: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_103, [1, 512, 1024]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_979: "f32[512, 1024]" = torch.ops.aten.view.default(view_978, [512, 1024]);  view_978 = None
    permute_789: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_190: "f32[512, 1024]" = torch.ops.aten.mm.default(view_979, permute_789);  permute_789 = None
    permute_790: "f32[1024, 512]" = torch.ops.aten.permute.default(view_979, [1, 0])
    mm_191: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_790, view_181);  permute_790 = view_181 = None
    permute_791: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_272: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
    view_980: "f32[1024]" = torch.ops.aten.view.default(sum_272, [1024]);  sum_272 = None
    permute_792: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_791, [1, 0]);  permute_791 = None
    view_981: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_190, [1, 512, 1024]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_793: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_786, [0, 2, 1, 3]);  permute_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_982: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_793, [1, 512, 1024]);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_983: "f32[512, 1024]" = torch.ops.aten.view.default(view_982, [512, 1024]);  view_982 = None
    permute_794: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_192: "f32[512, 1024]" = torch.ops.aten.mm.default(view_983, permute_794);  permute_794 = None
    permute_795: "f32[1024, 512]" = torch.ops.aten.permute.default(view_983, [1, 0])
    mm_193: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_795, view_178);  permute_795 = view_178 = None
    permute_796: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_273: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_983, [0], True);  view_983 = None
    view_984: "f32[1024]" = torch.ops.aten.view.default(sum_273, [1024]);  sum_273 = None
    permute_797: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_796, [1, 0]);  permute_796 = None
    view_985: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_192, [1, 512, 1024]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_295: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_981, view_985);  view_981 = view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_986: "f32[512, 1024]" = torch.ops.aten.view.default(view_977, [512, 1024]);  view_977 = None
    permute_798: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_194: "f32[512, 1024]" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
    permute_799: "f32[1024, 512]" = torch.ops.aten.permute.default(view_986, [1, 0])
    mm_195: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
    permute_800: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_274: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[1024]" = torch.ops.aten.view.default(sum_274, [1024]);  sum_274 = None
    permute_801: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
    view_988: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_194, [1, 512, 1024]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_296: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_295, view_988);  add_295 = view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_193: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  add_65 = getitem_83 = None
    mul_656: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_193, rsqrt_16);  sub_193 = None
    mul_657: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, primals_132);  primals_132 = None
    mul_658: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_657, 1024)
    sum_275: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True)
    mul_659: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_657, mul_656);  mul_657 = None
    sum_276: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True);  mul_659 = None
    mul_660: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_656, sum_276);  sum_276 = None
    sub_194: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_658, sum_275);  mul_658 = sum_275 = None
    sub_195: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_194, mul_660);  sub_194 = mul_660 = None
    div_99: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None
    mul_661: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_99, sub_195);  div_99 = sub_195 = None
    mul_662: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, mul_656);  mul_656 = None
    sum_277: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 1]);  mul_662 = None
    sum_278: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_297: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_294, mul_661);  add_294 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_49: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_663: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_664: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_297, mul_663);  mul_663 = None
    clone_104: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_664, memory_format = torch.contiguous_format);  mul_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_989: "f32[512, 1024]" = torch.ops.aten.view.default(clone_104, [512, 1024]);  clone_104 = None
    permute_802: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_196: "f32[512, 4096]" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
    permute_803: "f32[1024, 512]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_197: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
    permute_804: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_279: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[1024]" = torch.ops.aten.view.default(sum_279, [1024]);  sum_279 = None
    permute_805: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_804, [1, 0]);  permute_804 = None
    view_991: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_196, [1, 512, 4096]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_665: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_42: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_665);  mul_665 = None
    add_298: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_666: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_298, 0.5);  add_298 = None
    mul_667: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_668: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_667, -0.5);  mul_667 = None
    exp_43: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_668);  mul_668 = None
    mul_669: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_670: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, mul_669);  view_173 = mul_669 = None
    add_299: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_666, mul_670);  mul_666 = mul_670 = None
    mul_671: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_991, add_299);  view_991 = add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_992: "f32[512, 4096]" = torch.ops.aten.view.default(mul_671, [512, 4096]);  mul_671 = None
    permute_806: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_198: "f32[512, 1024]" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
    permute_807: "f32[4096, 512]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_199: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
    permute_808: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_280: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[4096]" = torch.ops.aten.view.default(sum_280, [4096]);  sum_280 = None
    permute_809: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_808, [1, 0]);  permute_808 = None
    view_994: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_198, [1, 512, 1024]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_196: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  add_61 = getitem_79 = None
    mul_672: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_196, rsqrt_15);  sub_196 = None
    mul_673: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, primals_126);  primals_126 = None
    mul_674: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_673, 1024)
    sum_281: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True)
    mul_675: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_673, mul_672);  mul_673 = None
    sum_282: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True);  mul_675 = None
    mul_676: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_672, sum_282);  sum_282 = None
    sub_197: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_674, sum_281);  mul_674 = sum_281 = None
    sub_198: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_197, mul_676);  sub_197 = mul_676 = None
    div_100: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None
    mul_677: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_100, sub_198);  div_100 = sub_198 = None
    mul_678: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, mul_672);  mul_672 = None
    sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 1]);  mul_678 = None
    sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_300: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_297, mul_677);  add_297 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_50: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_679: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_680: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_300, mul_679);  mul_679 = None
    clone_105: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_680, memory_format = torch.contiguous_format);  mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_995: "f32[512, 1024]" = torch.ops.aten.view.default(clone_105, [512, 1024]);  clone_105 = None
    permute_810: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_200: "f32[512, 1024]" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
    permute_811: "f32[1024, 512]" = torch.ops.aten.permute.default(view_995, [1, 0])
    mm_201: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
    permute_812: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_285: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
    view_996: "f32[1024]" = torch.ops.aten.view.default(sum_285, [1024]);  sum_285 = None
    permute_813: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_812, [1, 0]);  permute_812 = None
    view_997: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_200, [1, 512, 1024]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_998: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_997, [1, 512, 16, 64]);  view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_814: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_999: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_814, [16, 512, 64]);  permute_814 = None
    permute_815: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_112: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_815, view_999);  permute_815 = None
    permute_816: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_113: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_999, permute_816);  view_999 = permute_816 = None
    view_1000: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_112, [1, 16, 512, 64]);  bmm_112 = None
    view_1001: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_113, [1, 16, 512, 512]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_51: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_681: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_682: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1001, mul_681);  view_1001 = mul_681 = None
    clone_106: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_682, memory_format = torch.contiguous_format);  mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_42: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_683: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_106, alias_42);  clone_106 = None
    sum_286: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [-1], True)
    mul_684: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_42, sum_286);  alias_42 = sum_286 = None
    sub_199: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_101: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_199, 8.0);  sub_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1002: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_101, [16, 512, 512]);  div_101 = None
    permute_817: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_114: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_817, view_1002);  permute_817 = None
    permute_818: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    bmm_115: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1002, permute_818);  view_1002 = permute_818 = None
    view_1003: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_114, [1, 16, 64, 512]);  bmm_114 = None
    view_1004: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_115, [1, 16, 512, 64]);  bmm_115 = None
    permute_819: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1003, [0, 1, 3, 2]);  view_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_820: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1004, [0, 2, 1, 3]);  view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_107: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
    view_1005: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_107, [1, 512, 1024]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_821: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1000, [0, 2, 1, 3]);  view_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_108: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
    view_1006: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_108, [1, 512, 1024]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1007: "f32[512, 1024]" = torch.ops.aten.view.default(view_1006, [512, 1024]);  view_1006 = None
    permute_822: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_202: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1007, permute_822);  permute_822 = None
    permute_823: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_203: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_823, view_159);  permute_823 = view_159 = None
    permute_824: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_287: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
    view_1008: "f32[1024]" = torch.ops.aten.view.default(sum_287, [1024]);  sum_287 = None
    permute_825: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_1009: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_202, [1, 512, 1024]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_826: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_819, [0, 2, 1, 3]);  permute_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1010: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_826, [1, 512, 1024]);  permute_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1011: "f32[512, 1024]" = torch.ops.aten.view.default(view_1010, [512, 1024]);  view_1010 = None
    permute_827: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_204: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1011, permute_827);  permute_827 = None
    permute_828: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1011, [1, 0])
    mm_205: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_828, view_156);  permute_828 = view_156 = None
    permute_829: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_288: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1011, [0], True);  view_1011 = None
    view_1012: "f32[1024]" = torch.ops.aten.view.default(sum_288, [1024]);  sum_288 = None
    permute_830: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
    view_1013: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_204, [1, 512, 1024]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_301: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1009, view_1013);  view_1009 = view_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1014: "f32[512, 1024]" = torch.ops.aten.view.default(view_1005, [512, 1024]);  view_1005 = None
    permute_831: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_206: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
    permute_832: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_207: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
    permute_833: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_289: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[1024]" = torch.ops.aten.view.default(sum_289, [1024]);  sum_289 = None
    permute_834: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_833, [1, 0]);  permute_833 = None
    view_1016: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_206, [1, 512, 1024]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_302: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_301, view_1016);  add_301 = view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_200: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  add_57 = getitem_73 = None
    mul_685: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_14);  sub_200 = None
    mul_686: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, primals_116);  primals_116 = None
    mul_687: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, 1024)
    sum_290: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
    mul_688: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, mul_685);  mul_686 = None
    sum_291: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
    mul_689: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_685, sum_291);  sum_291 = None
    sub_201: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_687, sum_290);  mul_687 = sum_290 = None
    sub_202: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_201, mul_689);  sub_201 = mul_689 = None
    div_102: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None
    mul_690: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_102, sub_202);  div_102 = sub_202 = None
    mul_691: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_685);  mul_685 = None
    sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
    sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_303: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_300, mul_690);  add_300 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_52: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_692: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_693: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_303, mul_692);  mul_692 = None
    clone_109: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_693, memory_format = torch.contiguous_format);  mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1017: "f32[512, 1024]" = torch.ops.aten.view.default(clone_109, [512, 1024]);  clone_109 = None
    permute_835: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_208: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
    permute_836: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1017, [1, 0])
    mm_209: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
    permute_837: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
    view_1018: "f32[1024]" = torch.ops.aten.view.default(sum_294, [1024]);  sum_294 = None
    permute_838: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_837, [1, 0]);  permute_837 = None
    view_1019: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_208, [1, 512, 4096]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_694: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_43: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_694);  mul_694 = None
    add_304: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_695: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_304, 0.5);  add_304 = None
    mul_696: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_697: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_696, -0.5);  mul_696 = None
    exp_44: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_697);  mul_697 = None
    mul_698: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_699: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, mul_698);  view_151 = mul_698 = None
    add_305: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_695, mul_699);  mul_695 = mul_699 = None
    mul_700: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1019, add_305);  view_1019 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1020: "f32[512, 4096]" = torch.ops.aten.view.default(mul_700, [512, 4096]);  mul_700 = None
    permute_839: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_210: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
    permute_840: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1020, [1, 0])
    mm_211: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
    permute_841: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_295: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
    view_1021: "f32[4096]" = torch.ops.aten.view.default(sum_295, [4096]);  sum_295 = None
    permute_842: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    view_1022: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_210, [1, 512, 1024]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_203: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  add_53 = getitem_69 = None
    mul_701: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_203, rsqrt_13);  sub_203 = None
    mul_702: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, primals_110);  primals_110 = None
    mul_703: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, 1024)
    sum_296: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True)
    mul_704: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, mul_701);  mul_702 = None
    sum_297: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [2], True);  mul_704 = None
    mul_705: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_701, sum_297);  sum_297 = None
    sub_204: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_703, sum_296);  mul_703 = sum_296 = None
    sub_205: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_204, mul_705);  sub_204 = mul_705 = None
    div_103: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None
    mul_706: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_103, sub_205);  div_103 = sub_205 = None
    mul_707: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, mul_701);  mul_701 = None
    sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1]);  mul_707 = None
    sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1022, [0, 1]);  view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_306: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_303, mul_706);  add_303 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_53: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_708: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_709: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_306, mul_708);  mul_708 = None
    clone_110: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_709, memory_format = torch.contiguous_format);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1023: "f32[512, 1024]" = torch.ops.aten.view.default(clone_110, [512, 1024]);  clone_110 = None
    permute_843: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_212: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
    permute_844: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1023, [1, 0])
    mm_213: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
    permute_845: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_300: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
    view_1024: "f32[1024]" = torch.ops.aten.view.default(sum_300, [1024]);  sum_300 = None
    permute_846: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_845, [1, 0]);  permute_845 = None
    view_1025: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_212, [1, 512, 1024]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1026: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1025, [1, 512, 16, 64]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_847: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1027: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_847, [16, 512, 64]);  permute_847 = None
    permute_848: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_116: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_848, view_1027);  permute_848 = None
    permute_849: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    bmm_117: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1027, permute_849);  view_1027 = permute_849 = None
    view_1028: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_116, [1, 16, 512, 64]);  bmm_116 = None
    view_1029: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_117, [1, 16, 512, 512]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_54: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_710: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_711: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1029, mul_710);  view_1029 = mul_710 = None
    clone_111: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_711, memory_format = torch.contiguous_format);  mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_43: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_712: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_111, alias_43);  clone_111 = None
    sum_301: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [-1], True)
    mul_713: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_301);  alias_43 = sum_301 = None
    sub_206: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_104: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_206, 8.0);  sub_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1030: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_104, [16, 512, 512]);  div_104 = None
    permute_850: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_118: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_850, view_1030);  permute_850 = None
    permute_851: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_119: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1030, permute_851);  view_1030 = permute_851 = None
    view_1031: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_118, [1, 16, 64, 512]);  bmm_118 = None
    view_1032: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_119, [1, 16, 512, 64]);  bmm_119 = None
    permute_852: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1031, [0, 1, 3, 2]);  view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_853: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1032, [0, 2, 1, 3]);  view_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_112: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_853, memory_format = torch.contiguous_format);  permute_853 = None
    view_1033: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_112, [1, 512, 1024]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_854: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1028, [0, 2, 1, 3]);  view_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_113: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_854, memory_format = torch.contiguous_format);  permute_854 = None
    view_1034: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_113, [1, 512, 1024]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1035: "f32[512, 1024]" = torch.ops.aten.view.default(view_1034, [512, 1024]);  view_1034 = None
    permute_855: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_214: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1035, permute_855);  permute_855 = None
    permute_856: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1035, [1, 0])
    mm_215: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_856, view_137);  permute_856 = view_137 = None
    permute_857: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1035, [0], True);  view_1035 = None
    view_1036: "f32[1024]" = torch.ops.aten.view.default(sum_302, [1024]);  sum_302 = None
    permute_858: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_857, [1, 0]);  permute_857 = None
    view_1037: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_214, [1, 512, 1024]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_859: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_852, [0, 2, 1, 3]);  permute_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1038: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_859, [1, 512, 1024]);  permute_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1039: "f32[512, 1024]" = torch.ops.aten.view.default(view_1038, [512, 1024]);  view_1038 = None
    permute_860: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_216: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1039, permute_860);  permute_860 = None
    permute_861: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1039, [1, 0])
    mm_217: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_861, view_134);  permute_861 = view_134 = None
    permute_862: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1039, [0], True);  view_1039 = None
    view_1040: "f32[1024]" = torch.ops.aten.view.default(sum_303, [1024]);  sum_303 = None
    permute_863: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_862, [1, 0]);  permute_862 = None
    view_1041: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_216, [1, 512, 1024]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_307: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1037, view_1041);  view_1037 = view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1042: "f32[512, 1024]" = torch.ops.aten.view.default(view_1033, [512, 1024]);  view_1033 = None
    permute_864: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_218: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
    permute_865: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1042, [1, 0])
    mm_219: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
    permute_866: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_304: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
    view_1043: "f32[1024]" = torch.ops.aten.view.default(sum_304, [1024]);  sum_304 = None
    permute_867: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
    view_1044: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_218, [1, 512, 1024]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_308: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_307, view_1044);  add_307 = view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_207: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  add_49 = getitem_63 = None
    mul_714: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_207, rsqrt_12);  sub_207 = None
    mul_715: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, primals_100);  primals_100 = None
    mul_716: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_715, 1024)
    sum_305: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True)
    mul_717: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_715, mul_714);  mul_715 = None
    sum_306: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True);  mul_717 = None
    mul_718: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_714, sum_306);  sum_306 = None
    sub_208: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_716, sum_305);  mul_716 = sum_305 = None
    sub_209: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_208, mul_718);  sub_208 = mul_718 = None
    div_105: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None
    mul_719: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_105, sub_209);  div_105 = sub_209 = None
    mul_720: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_714);  mul_714 = None
    sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 1]);  mul_720 = None
    sum_308: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_309: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_306, mul_719);  add_306 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_55: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_721: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_722: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_309, mul_721);  mul_721 = None
    clone_114: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_722, memory_format = torch.contiguous_format);  mul_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1045: "f32[512, 1024]" = torch.ops.aten.view.default(clone_114, [512, 1024]);  clone_114 = None
    permute_868: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_220: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
    permute_869: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1045, [1, 0])
    mm_221: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
    permute_870: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
    view_1046: "f32[1024]" = torch.ops.aten.view.default(sum_309, [1024]);  sum_309 = None
    permute_871: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_870, [1, 0]);  permute_870 = None
    view_1047: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_220, [1, 512, 4096]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_723: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_44: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_723);  mul_723 = None
    add_310: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_724: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_310, 0.5);  add_310 = None
    mul_725: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_726: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_725, -0.5);  mul_725 = None
    exp_45: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_726);  mul_726 = None
    mul_727: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_728: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, mul_727);  view_129 = mul_727 = None
    add_311: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_724, mul_728);  mul_724 = mul_728 = None
    mul_729: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1047, add_311);  view_1047 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1048: "f32[512, 4096]" = torch.ops.aten.view.default(mul_729, [512, 4096]);  mul_729 = None
    permute_872: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_222: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
    permute_873: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1048, [1, 0])
    mm_223: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
    permute_874: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_310: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
    view_1049: "f32[4096]" = torch.ops.aten.view.default(sum_310, [4096]);  sum_310 = None
    permute_875: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_874, [1, 0]);  permute_874 = None
    view_1050: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_222, [1, 512, 1024]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_210: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  add_45 = getitem_59 = None
    mul_730: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_210, rsqrt_11);  sub_210 = None
    mul_731: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, primals_94);  primals_94 = None
    mul_732: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, 1024)
    sum_311: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
    mul_733: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, mul_730);  mul_731 = None
    sum_312: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
    mul_734: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_730, sum_312);  sum_312 = None
    sub_211: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_732, sum_311);  mul_732 = sum_311 = None
    sub_212: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_211, mul_734);  sub_211 = mul_734 = None
    div_106: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None
    mul_735: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_106, sub_212);  div_106 = sub_212 = None
    mul_736: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, mul_730);  mul_730 = None
    sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
    sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1050, [0, 1]);  view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_312: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_309, mul_735);  add_309 = mul_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_56: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_737: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_738: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_312, mul_737);  mul_737 = None
    clone_115: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_738, memory_format = torch.contiguous_format);  mul_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1051: "f32[512, 1024]" = torch.ops.aten.view.default(clone_115, [512, 1024]);  clone_115 = None
    permute_876: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_224: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
    permute_877: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1051, [1, 0])
    mm_225: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
    permute_878: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_315: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
    view_1052: "f32[1024]" = torch.ops.aten.view.default(sum_315, [1024]);  sum_315 = None
    permute_879: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_878, [1, 0]);  permute_878 = None
    view_1053: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_224, [1, 512, 1024]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1054: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1053, [1, 512, 16, 64]);  view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_880: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1055: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_880, [16, 512, 64]);  permute_880 = None
    permute_881: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_120: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_881, view_1055);  permute_881 = None
    permute_882: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_121: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1055, permute_882);  view_1055 = permute_882 = None
    view_1056: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_120, [1, 16, 512, 64]);  bmm_120 = None
    view_1057: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_121, [1, 16, 512, 512]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_57: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_739: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_740: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1057, mul_739);  view_1057 = mul_739 = None
    clone_116: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_740, memory_format = torch.contiguous_format);  mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_44: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_741: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_116, alias_44);  clone_116 = None
    sum_316: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [-1], True)
    mul_742: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_44, sum_316);  alias_44 = sum_316 = None
    sub_213: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_107: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_213, 8.0);  sub_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1058: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_107, [16, 512, 512]);  div_107 = None
    permute_883: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    bmm_122: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_883, view_1058);  permute_883 = None
    permute_884: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    bmm_123: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1058, permute_884);  view_1058 = permute_884 = None
    view_1059: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_122, [1, 16, 64, 512]);  bmm_122 = None
    view_1060: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_123, [1, 16, 512, 64]);  bmm_123 = None
    permute_885: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1059, [0, 1, 3, 2]);  view_1059 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_886: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1060, [0, 2, 1, 3]);  view_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_117: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
    view_1061: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_117, [1, 512, 1024]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_887: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1056, [0, 2, 1, 3]);  view_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_118: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
    view_1062: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_118, [1, 512, 1024]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1063: "f32[512, 1024]" = torch.ops.aten.view.default(view_1062, [512, 1024]);  view_1062 = None
    permute_888: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_226: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1063, permute_888);  permute_888 = None
    permute_889: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1063, [1, 0])
    mm_227: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_889, view_115);  permute_889 = view_115 = None
    permute_890: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1063, [0], True);  view_1063 = None
    view_1064: "f32[1024]" = torch.ops.aten.view.default(sum_317, [1024]);  sum_317 = None
    permute_891: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_890, [1, 0]);  permute_890 = None
    view_1065: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_226, [1, 512, 1024]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_892: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_885, [0, 2, 1, 3]);  permute_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1066: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_892, [1, 512, 1024]);  permute_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1067: "f32[512, 1024]" = torch.ops.aten.view.default(view_1066, [512, 1024]);  view_1066 = None
    permute_893: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_228: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1067, permute_893);  permute_893 = None
    permute_894: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1067, [1, 0])
    mm_229: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_894, view_112);  permute_894 = view_112 = None
    permute_895: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1067, [0], True);  view_1067 = None
    view_1068: "f32[1024]" = torch.ops.aten.view.default(sum_318, [1024]);  sum_318 = None
    permute_896: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_895, [1, 0]);  permute_895 = None
    view_1069: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_228, [1, 512, 1024]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_313: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1065, view_1069);  view_1065 = view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1070: "f32[512, 1024]" = torch.ops.aten.view.default(view_1061, [512, 1024]);  view_1061 = None
    permute_897: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_230: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
    permute_898: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1070, [1, 0])
    mm_231: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
    permute_899: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
    view_1071: "f32[1024]" = torch.ops.aten.view.default(sum_319, [1024]);  sum_319 = None
    permute_900: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_899, [1, 0]);  permute_899 = None
    view_1072: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_230, [1, 512, 1024]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_314: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_313, view_1072);  add_313 = view_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_214: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  add_41 = getitem_53 = None
    mul_743: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_214, rsqrt_10);  sub_214 = None
    mul_744: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, primals_84);  primals_84 = None
    mul_745: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_744, 1024)
    sum_320: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True)
    mul_746: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_744, mul_743);  mul_744 = None
    sum_321: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True);  mul_746 = None
    mul_747: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_743, sum_321);  sum_321 = None
    sub_215: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_745, sum_320);  mul_745 = sum_320 = None
    sub_216: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_215, mul_747);  sub_215 = mul_747 = None
    div_108: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
    mul_748: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_108, sub_216);  div_108 = sub_216 = None
    mul_749: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, mul_743);  mul_743 = None
    sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1]);  mul_749 = None
    sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_315: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_312, mul_748);  add_312 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_58: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_750: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_751: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_315, mul_750);  mul_750 = None
    clone_119: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_751, memory_format = torch.contiguous_format);  mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1073: "f32[512, 1024]" = torch.ops.aten.view.default(clone_119, [512, 1024]);  clone_119 = None
    permute_901: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_232: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
    permute_902: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1073, [1, 0])
    mm_233: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
    permute_903: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_324: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
    view_1074: "f32[1024]" = torch.ops.aten.view.default(sum_324, [1024]);  sum_324 = None
    permute_904: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
    view_1075: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_232, [1, 512, 4096]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_752: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_45: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_752);  mul_752 = None
    add_316: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_753: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_316, 0.5);  add_316 = None
    mul_754: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_755: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_754, -0.5);  mul_754 = None
    exp_46: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_755);  mul_755 = None
    mul_756: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_757: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, mul_756);  view_107 = mul_756 = None
    add_317: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_753, mul_757);  mul_753 = mul_757 = None
    mul_758: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1075, add_317);  view_1075 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1076: "f32[512, 4096]" = torch.ops.aten.view.default(mul_758, [512, 4096]);  mul_758 = None
    permute_905: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_234: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
    permute_906: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1076, [1, 0])
    mm_235: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
    permute_907: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_325: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
    view_1077: "f32[4096]" = torch.ops.aten.view.default(sum_325, [4096]);  sum_325 = None
    permute_908: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_907, [1, 0]);  permute_907 = None
    view_1078: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_234, [1, 512, 1024]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_217: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  add_37 = getitem_49 = None
    mul_759: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_9);  sub_217 = None
    mul_760: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, primals_78);  primals_78 = None
    mul_761: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_760, 1024)
    sum_326: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [2], True)
    mul_762: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_760, mul_759);  mul_760 = None
    sum_327: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_762, [2], True);  mul_762 = None
    mul_763: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_759, sum_327);  sum_327 = None
    sub_218: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_761, sum_326);  mul_761 = sum_326 = None
    sub_219: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_218, mul_763);  sub_218 = mul_763 = None
    div_109: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None
    mul_764: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_109, sub_219);  div_109 = sub_219 = None
    mul_765: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, mul_759);  mul_759 = None
    sum_328: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_765, [0, 1]);  mul_765 = None
    sum_329: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1078, [0, 1]);  view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_318: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_315, mul_764);  add_315 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_59: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_766: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_767: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_318, mul_766);  mul_766 = None
    clone_120: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_767, memory_format = torch.contiguous_format);  mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1079: "f32[512, 1024]" = torch.ops.aten.view.default(clone_120, [512, 1024]);  clone_120 = None
    permute_909: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_236: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
    permute_910: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1079, [1, 0])
    mm_237: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
    permute_911: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_330: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
    view_1080: "f32[1024]" = torch.ops.aten.view.default(sum_330, [1024]);  sum_330 = None
    permute_912: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_911, [1, 0]);  permute_911 = None
    view_1081: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_236, [1, 512, 1024]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1082: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1081, [1, 512, 16, 64]);  view_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_913: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1083: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_913, [16, 512, 64]);  permute_913 = None
    permute_914: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_124: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_914, view_1083);  permute_914 = None
    permute_915: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_125: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1083, permute_915);  view_1083 = permute_915 = None
    view_1084: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_124, [1, 16, 512, 64]);  bmm_124 = None
    view_1085: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_125, [1, 16, 512, 512]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_60: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_768: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_769: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1085, mul_768);  view_1085 = mul_768 = None
    clone_121: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_769, memory_format = torch.contiguous_format);  mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_45: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_770: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_121, alias_45);  clone_121 = None
    sum_331: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [-1], True)
    mul_771: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_331);  alias_45 = sum_331 = None
    sub_220: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_770, mul_771);  mul_770 = mul_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_110: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_220, 8.0);  sub_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1086: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_110, [16, 512, 512]);  div_110 = None
    permute_916: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_126: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_916, view_1086);  permute_916 = None
    permute_917: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_127: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1086, permute_917);  view_1086 = permute_917 = None
    view_1087: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_126, [1, 16, 64, 512]);  bmm_126 = None
    view_1088: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_127, [1, 16, 512, 64]);  bmm_127 = None
    permute_918: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1087, [0, 1, 3, 2]);  view_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_919: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1088, [0, 2, 1, 3]);  view_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_122: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
    view_1089: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_122, [1, 512, 1024]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_920: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1084, [0, 2, 1, 3]);  view_1084 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_123: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_920, memory_format = torch.contiguous_format);  permute_920 = None
    view_1090: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_123, [1, 512, 1024]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1091: "f32[512, 1024]" = torch.ops.aten.view.default(view_1090, [512, 1024]);  view_1090 = None
    permute_921: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_238: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1091, permute_921);  permute_921 = None
    permute_922: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1091, [1, 0])
    mm_239: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_922, view_93);  permute_922 = view_93 = None
    permute_923: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_332: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1091, [0], True);  view_1091 = None
    view_1092: "f32[1024]" = torch.ops.aten.view.default(sum_332, [1024]);  sum_332 = None
    permute_924: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_923, [1, 0]);  permute_923 = None
    view_1093: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_238, [1, 512, 1024]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_925: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_918, [0, 2, 1, 3]);  permute_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1094: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_925, [1, 512, 1024]);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1095: "f32[512, 1024]" = torch.ops.aten.view.default(view_1094, [512, 1024]);  view_1094 = None
    permute_926: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_240: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1095, permute_926);  permute_926 = None
    permute_927: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1095, [1, 0])
    mm_241: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_927, view_90);  permute_927 = view_90 = None
    permute_928: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1095, [0], True);  view_1095 = None
    view_1096: "f32[1024]" = torch.ops.aten.view.default(sum_333, [1024]);  sum_333 = None
    permute_929: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_928, [1, 0]);  permute_928 = None
    view_1097: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_240, [1, 512, 1024]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_319: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1093, view_1097);  view_1093 = view_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1098: "f32[512, 1024]" = torch.ops.aten.view.default(view_1089, [512, 1024]);  view_1089 = None
    permute_930: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_242: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
    permute_931: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1098, [1, 0])
    mm_243: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
    permute_932: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_334: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
    view_1099: "f32[1024]" = torch.ops.aten.view.default(sum_334, [1024]);  sum_334 = None
    permute_933: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_932, [1, 0]);  permute_932 = None
    view_1100: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_242, [1, 512, 1024]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_320: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_319, view_1100);  add_319 = view_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_221: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  add_33 = getitem_43 = None
    mul_772: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_221, rsqrt_8);  sub_221 = None
    mul_773: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, primals_68);  primals_68 = None
    mul_774: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_773, 1024)
    sum_335: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True)
    mul_775: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_773, mul_772);  mul_773 = None
    sum_336: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
    mul_776: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_772, sum_336);  sum_336 = None
    sub_222: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_774, sum_335);  mul_774 = sum_335 = None
    sub_223: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_222, mul_776);  sub_222 = mul_776 = None
    div_111: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None
    mul_777: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_111, sub_223);  div_111 = sub_223 = None
    mul_778: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_772);  mul_772 = None
    sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 1]);  mul_778 = None
    sum_338: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_321: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_318, mul_777);  add_318 = mul_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_61: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_779: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_780: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_321, mul_779);  mul_779 = None
    clone_124: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_780, memory_format = torch.contiguous_format);  mul_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1101: "f32[512, 1024]" = torch.ops.aten.view.default(clone_124, [512, 1024]);  clone_124 = None
    permute_934: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_244: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
    permute_935: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1101, [1, 0])
    mm_245: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
    permute_936: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_339: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
    view_1102: "f32[1024]" = torch.ops.aten.view.default(sum_339, [1024]);  sum_339 = None
    permute_937: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_936, [1, 0]);  permute_936 = None
    view_1103: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_244, [1, 512, 4096]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_781: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_46: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_781);  mul_781 = None
    add_322: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_782: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_322, 0.5);  add_322 = None
    mul_783: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_784: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_783, -0.5);  mul_783 = None
    exp_47: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_784);  mul_784 = None
    mul_785: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_786: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, mul_785);  view_85 = mul_785 = None
    add_323: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_782, mul_786);  mul_782 = mul_786 = None
    mul_787: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1103, add_323);  view_1103 = add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1104: "f32[512, 4096]" = torch.ops.aten.view.default(mul_787, [512, 4096]);  mul_787 = None
    permute_938: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_246: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
    permute_939: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1104, [1, 0])
    mm_247: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
    permute_940: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_340: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
    view_1105: "f32[4096]" = torch.ops.aten.view.default(sum_340, [4096]);  sum_340 = None
    permute_941: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_940, [1, 0]);  permute_940 = None
    view_1106: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_246, [1, 512, 1024]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_224: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  add_29 = getitem_39 = None
    mul_788: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_224, rsqrt_7);  sub_224 = None
    mul_789: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, primals_62);  primals_62 = None
    mul_790: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_789, 1024)
    sum_341: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
    mul_791: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_789, mul_788);  mul_789 = None
    sum_342: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
    mul_792: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_788, sum_342);  sum_342 = None
    sub_225: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_790, sum_341);  mul_790 = sum_341 = None
    sub_226: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_225, mul_792);  sub_225 = mul_792 = None
    div_112: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None
    mul_793: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_112, sub_226);  div_112 = sub_226 = None
    mul_794: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, mul_788);  mul_788 = None
    sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
    sum_344: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1106, [0, 1]);  view_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_324: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_321, mul_793);  add_321 = mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_62: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_795: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_796: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_324, mul_795);  mul_795 = None
    clone_125: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_796, memory_format = torch.contiguous_format);  mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1107: "f32[512, 1024]" = torch.ops.aten.view.default(clone_125, [512, 1024]);  clone_125 = None
    permute_942: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_248: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
    permute_943: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1107, [1, 0])
    mm_249: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
    permute_944: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_345: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
    view_1108: "f32[1024]" = torch.ops.aten.view.default(sum_345, [1024]);  sum_345 = None
    permute_945: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_944, [1, 0]);  permute_944 = None
    view_1109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_248, [1, 512, 1024]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1110: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1109, [1, 512, 16, 64]);  view_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_946: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1111: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_946, [16, 512, 64]);  permute_946 = None
    permute_947: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_128: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_947, view_1111);  permute_947 = None
    permute_948: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_129: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1111, permute_948);  view_1111 = permute_948 = None
    view_1112: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_128, [1, 16, 512, 64]);  bmm_128 = None
    view_1113: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_129, [1, 16, 512, 512]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_63: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_797: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_798: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1113, mul_797);  view_1113 = mul_797 = None
    clone_126: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_798, memory_format = torch.contiguous_format);  mul_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_46: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_799: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_126, alias_46);  clone_126 = None
    sum_346: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_799, [-1], True)
    mul_800: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_46, sum_346);  alias_46 = sum_346 = None
    sub_227: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_113: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_227, 8.0);  sub_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1114: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_113, [16, 512, 512]);  div_113 = None
    permute_949: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_130: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_949, view_1114);  permute_949 = None
    permute_950: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_131: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1114, permute_950);  view_1114 = permute_950 = None
    view_1115: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_130, [1, 16, 64, 512]);  bmm_130 = None
    view_1116: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_131, [1, 16, 512, 64]);  bmm_131 = None
    permute_951: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1115, [0, 1, 3, 2]);  view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_952: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1116, [0, 2, 1, 3]);  view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_127: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
    view_1117: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_127, [1, 512, 1024]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_953: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1112, [0, 2, 1, 3]);  view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_128: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_953, memory_format = torch.contiguous_format);  permute_953 = None
    view_1118: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_128, [1, 512, 1024]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1119: "f32[512, 1024]" = torch.ops.aten.view.default(view_1118, [512, 1024]);  view_1118 = None
    permute_954: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_250: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1119, permute_954);  permute_954 = None
    permute_955: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1119, [1, 0])
    mm_251: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_955, view_71);  permute_955 = view_71 = None
    permute_956: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1119, [0], True);  view_1119 = None
    view_1120: "f32[1024]" = torch.ops.aten.view.default(sum_347, [1024]);  sum_347 = None
    permute_957: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_956, [1, 0]);  permute_956 = None
    view_1121: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_250, [1, 512, 1024]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_958: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_951, [0, 2, 1, 3]);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1122: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_958, [1, 512, 1024]);  permute_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1123: "f32[512, 1024]" = torch.ops.aten.view.default(view_1122, [512, 1024]);  view_1122 = None
    permute_959: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_252: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1123, permute_959);  permute_959 = None
    permute_960: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1123, [1, 0])
    mm_253: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_960, view_68);  permute_960 = view_68 = None
    permute_961: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_348: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1123, [0], True);  view_1123 = None
    view_1124: "f32[1024]" = torch.ops.aten.view.default(sum_348, [1024]);  sum_348 = None
    permute_962: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_961, [1, 0]);  permute_961 = None
    view_1125: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_252, [1, 512, 1024]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_325: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1121, view_1125);  view_1121 = view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1126: "f32[512, 1024]" = torch.ops.aten.view.default(view_1117, [512, 1024]);  view_1117 = None
    permute_963: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_254: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
    permute_964: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1126, [1, 0])
    mm_255: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
    permute_965: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_349: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
    view_1127: "f32[1024]" = torch.ops.aten.view.default(sum_349, [1024]);  sum_349 = None
    permute_966: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_965, [1, 0]);  permute_965 = None
    view_1128: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_254, [1, 512, 1024]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_326: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_325, view_1128);  add_325 = view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_228: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  add_25 = getitem_33 = None
    mul_801: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_228, rsqrt_6);  sub_228 = None
    mul_802: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, primals_52);  primals_52 = None
    mul_803: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_802, 1024)
    sum_350: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_802, [2], True)
    mul_804: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_802, mul_801);  mul_802 = None
    sum_351: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True);  mul_804 = None
    mul_805: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_801, sum_351);  sum_351 = None
    sub_229: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_803, sum_350);  mul_803 = sum_350 = None
    sub_230: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_229, mul_805);  sub_229 = mul_805 = None
    div_114: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None
    mul_806: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_114, sub_230);  div_114 = sub_230 = None
    mul_807: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, mul_801);  mul_801 = None
    sum_352: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 1]);  mul_807 = None
    sum_353: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_327: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_324, mul_806);  add_324 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_808: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_809: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_327, mul_808);  mul_808 = None
    clone_129: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_809, memory_format = torch.contiguous_format);  mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1129: "f32[512, 1024]" = torch.ops.aten.view.default(clone_129, [512, 1024]);  clone_129 = None
    permute_967: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_256: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
    permute_968: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1129, [1, 0])
    mm_257: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
    permute_969: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_354: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
    view_1130: "f32[1024]" = torch.ops.aten.view.default(sum_354, [1024]);  sum_354 = None
    permute_970: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_969, [1, 0]);  permute_969 = None
    view_1131: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_256, [1, 512, 4096]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_810: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_47: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_810);  mul_810 = None
    add_328: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_811: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_328, 0.5);  add_328 = None
    mul_812: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_813: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_48: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_815: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, mul_814);  view_63 = mul_814 = None
    add_329: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1131, add_329);  view_1131 = add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1132: "f32[512, 4096]" = torch.ops.aten.view.default(mul_816, [512, 4096]);  mul_816 = None
    permute_971: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_258: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
    permute_972: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1132, [1, 0])
    mm_259: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
    permute_973: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_355: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
    view_1133: "f32[4096]" = torch.ops.aten.view.default(sum_355, [4096]);  sum_355 = None
    permute_974: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_973, [1, 0]);  permute_973 = None
    view_1134: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_258, [1, 512, 1024]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_231: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  add_21 = getitem_29 = None
    mul_817: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_231, rsqrt_5);  sub_231 = None
    mul_818: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, primals_46);  primals_46 = None
    mul_819: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_818, 1024)
    sum_356: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [2], True)
    mul_820: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_818, mul_817);  mul_818 = None
    sum_357: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
    mul_821: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_817, sum_357);  sum_357 = None
    sub_232: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_819, sum_356);  mul_819 = sum_356 = None
    sub_233: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_232, mul_821);  sub_232 = mul_821 = None
    div_115: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None
    mul_822: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_115, sub_233);  div_115 = sub_233 = None
    mul_823: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, mul_817);  mul_817 = None
    sum_358: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
    sum_359: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1134, [0, 1]);  view_1134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_330: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_327, mul_822);  add_327 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_65: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_824: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_825: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_330, mul_824);  mul_824 = None
    clone_130: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_825, memory_format = torch.contiguous_format);  mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1135: "f32[512, 1024]" = torch.ops.aten.view.default(clone_130, [512, 1024]);  clone_130 = None
    permute_975: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_260: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
    permute_976: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1135, [1, 0])
    mm_261: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
    permute_977: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_360: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
    view_1136: "f32[1024]" = torch.ops.aten.view.default(sum_360, [1024]);  sum_360 = None
    permute_978: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_977, [1, 0]);  permute_977 = None
    view_1137: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_260, [1, 512, 1024]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1138: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1137, [1, 512, 16, 64]);  view_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_979: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1139: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_979, [16, 512, 64]);  permute_979 = None
    permute_980: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_132: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_980, view_1139);  permute_980 = None
    permute_981: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_133: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1139, permute_981);  view_1139 = permute_981 = None
    view_1140: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_132, [1, 16, 512, 64]);  bmm_132 = None
    view_1141: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_133, [1, 16, 512, 512]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_66: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_826: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_827: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1141, mul_826);  view_1141 = mul_826 = None
    clone_131: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_827, memory_format = torch.contiguous_format);  mul_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_47: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_828: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_131, alias_47);  clone_131 = None
    sum_361: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_828, [-1], True)
    mul_829: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_47, sum_361);  alias_47 = sum_361 = None
    sub_234: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_116: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_234, 8.0);  sub_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1142: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_116, [16, 512, 512]);  div_116 = None
    permute_982: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_134: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_982, view_1142);  permute_982 = None
    permute_983: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_135: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1142, permute_983);  view_1142 = permute_983 = None
    view_1143: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_134, [1, 16, 64, 512]);  bmm_134 = None
    view_1144: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_135, [1, 16, 512, 64]);  bmm_135 = None
    permute_984: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1143, [0, 1, 3, 2]);  view_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_985: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1144, [0, 2, 1, 3]);  view_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_132: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
    view_1145: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_132, [1, 512, 1024]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_986: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1140, [0, 2, 1, 3]);  view_1140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_133: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_986, memory_format = torch.contiguous_format);  permute_986 = None
    view_1146: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_133, [1, 512, 1024]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1147: "f32[512, 1024]" = torch.ops.aten.view.default(view_1146, [512, 1024]);  view_1146 = None
    permute_987: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_262: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1147, permute_987);  permute_987 = None
    permute_988: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1147, [1, 0])
    mm_263: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_988, view_49);  permute_988 = view_49 = None
    permute_989: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1147, [0], True);  view_1147 = None
    view_1148: "f32[1024]" = torch.ops.aten.view.default(sum_362, [1024]);  sum_362 = None
    permute_990: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_989, [1, 0]);  permute_989 = None
    view_1149: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_262, [1, 512, 1024]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_991: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_984, [0, 2, 1, 3]);  permute_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1150: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_991, [1, 512, 1024]);  permute_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1151: "f32[512, 1024]" = torch.ops.aten.view.default(view_1150, [512, 1024]);  view_1150 = None
    permute_992: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_264: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1151, permute_992);  permute_992 = None
    permute_993: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1151, [1, 0])
    mm_265: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_993, view_46);  permute_993 = view_46 = None
    permute_994: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_363: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True);  view_1151 = None
    view_1152: "f32[1024]" = torch.ops.aten.view.default(sum_363, [1024]);  sum_363 = None
    permute_995: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_994, [1, 0]);  permute_994 = None
    view_1153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_264, [1, 512, 1024]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_331: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1149, view_1153);  view_1149 = view_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1154: "f32[512, 1024]" = torch.ops.aten.view.default(view_1145, [512, 1024]);  view_1145 = None
    permute_996: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_266: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
    permute_997: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1154, [1, 0])
    mm_267: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
    permute_998: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_364: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
    view_1155: "f32[1024]" = torch.ops.aten.view.default(sum_364, [1024]);  sum_364 = None
    permute_999: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_998, [1, 0]);  permute_998 = None
    view_1156: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_266, [1, 512, 1024]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_332: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_331, view_1156);  add_331 = view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_235: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_830: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_235, rsqrt_4);  sub_235 = None
    mul_831: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, primals_36);  primals_36 = None
    mul_832: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, 1024)
    sum_365: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
    mul_833: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, mul_830);  mul_831 = None
    sum_366: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
    mul_834: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_830, sum_366);  sum_366 = None
    sub_236: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_832, sum_365);  mul_832 = sum_365 = None
    sub_237: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_236, mul_834);  sub_236 = mul_834 = None
    div_117: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None
    mul_835: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_117, sub_237);  div_117 = sub_237 = None
    mul_836: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_830);  mul_830 = None
    sum_367: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
    sum_368: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_333: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_330, mul_835);  add_330 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_67: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_837: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_838: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_333, mul_837);  mul_837 = None
    clone_134: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_838, memory_format = torch.contiguous_format);  mul_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1157: "f32[512, 1024]" = torch.ops.aten.view.default(clone_134, [512, 1024]);  clone_134 = None
    permute_1000: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_268: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
    permute_1001: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1157, [1, 0])
    mm_269: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
    permute_1002: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_369: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
    view_1158: "f32[1024]" = torch.ops.aten.view.default(sum_369, [1024]);  sum_369 = None
    permute_1003: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1002, [1, 0]);  permute_1002 = None
    view_1159: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_268, [1, 512, 4096]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_839: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_48: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_839);  mul_839 = None
    add_334: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_840: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_334, 0.5);  add_334 = None
    mul_841: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_842: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_841, -0.5);  mul_841 = None
    exp_49: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_842);  mul_842 = None
    mul_843: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_844: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, mul_843);  view_41 = mul_843 = None
    add_335: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_840, mul_844);  mul_840 = mul_844 = None
    mul_845: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1159, add_335);  view_1159 = add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1160: "f32[512, 4096]" = torch.ops.aten.view.default(mul_845, [512, 4096]);  mul_845 = None
    permute_1004: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_270: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
    permute_1005: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1160, [1, 0])
    mm_271: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
    permute_1006: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_370: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
    view_1161: "f32[4096]" = torch.ops.aten.view.default(sum_370, [4096]);  sum_370 = None
    permute_1007: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1006, [1, 0]);  permute_1006 = None
    view_1162: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_270, [1, 512, 1024]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_238: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  add_13 = getitem_19 = None
    mul_846: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_238, rsqrt_3);  sub_238 = None
    mul_847: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, primals_30);  primals_30 = None
    mul_848: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_847, 1024)
    sum_371: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_847, [2], True)
    mul_849: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_847, mul_846);  mul_847 = None
    sum_372: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_849, [2], True);  mul_849 = None
    mul_850: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_846, sum_372);  sum_372 = None
    sub_239: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_848, sum_371);  mul_848 = sum_371 = None
    sub_240: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_239, mul_850);  sub_239 = mul_850 = None
    div_118: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
    mul_851: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_118, sub_240);  div_118 = sub_240 = None
    mul_852: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, mul_846);  mul_846 = None
    sum_373: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 1]);  mul_852 = None
    sum_374: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1162, [0, 1]);  view_1162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_336: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_333, mul_851);  add_333 = mul_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_853: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_854: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_336, mul_853);  mul_853 = None
    clone_135: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_854, memory_format = torch.contiguous_format);  mul_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1163: "f32[512, 1024]" = torch.ops.aten.view.default(clone_135, [512, 1024]);  clone_135 = None
    permute_1008: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_272: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
    permute_1009: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1163, [1, 0])
    mm_273: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
    permute_1010: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_375: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
    view_1164: "f32[1024]" = torch.ops.aten.view.default(sum_375, [1024]);  sum_375 = None
    permute_1011: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1010, [1, 0]);  permute_1010 = None
    view_1165: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_272, [1, 512, 1024]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1166: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1165, [1, 512, 16, 64]);  view_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1012: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1167: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1012, [16, 512, 64]);  permute_1012 = None
    permute_1013: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_136: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1013, view_1167);  permute_1013 = None
    permute_1014: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_137: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1167, permute_1014);  view_1167 = permute_1014 = None
    view_1168: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_136, [1, 16, 512, 64]);  bmm_136 = None
    view_1169: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_137, [1, 16, 512, 512]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_69: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_855: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_856: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1169, mul_855);  view_1169 = mul_855 = None
    clone_136: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_856, memory_format = torch.contiguous_format);  mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_48: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_857: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_136, alias_48);  clone_136 = None
    sum_376: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_857, [-1], True)
    mul_858: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_376);  alias_48 = sum_376 = None
    sub_241: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_857, mul_858);  mul_857 = mul_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_119: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_241, 8.0);  sub_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1170: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_119, [16, 512, 512]);  div_119 = None
    permute_1015: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_138: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1015, view_1170);  permute_1015 = None
    permute_1016: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_139: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1170, permute_1016);  view_1170 = permute_1016 = None
    view_1171: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_138, [1, 16, 64, 512]);  bmm_138 = None
    view_1172: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_139, [1, 16, 512, 64]);  bmm_139 = None
    permute_1017: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1171, [0, 1, 3, 2]);  view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1018: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1172, [0, 2, 1, 3]);  view_1172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_137: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1018, memory_format = torch.contiguous_format);  permute_1018 = None
    view_1173: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_137, [1, 512, 1024]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1019: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1168, [0, 2, 1, 3]);  view_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_138: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1019, memory_format = torch.contiguous_format);  permute_1019 = None
    view_1174: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_138, [1, 512, 1024]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1175: "f32[512, 1024]" = torch.ops.aten.view.default(view_1174, [512, 1024]);  view_1174 = None
    permute_1020: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_274: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1175, permute_1020);  permute_1020 = None
    permute_1021: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1175, [1, 0])
    mm_275: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1021, view_27);  permute_1021 = view_27 = None
    permute_1022: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1175, [0], True);  view_1175 = None
    view_1176: "f32[1024]" = torch.ops.aten.view.default(sum_377, [1024]);  sum_377 = None
    permute_1023: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1022, [1, 0]);  permute_1022 = None
    view_1177: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_274, [1, 512, 1024]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1024: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_1017, [0, 2, 1, 3]);  permute_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1178: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_1024, [1, 512, 1024]);  permute_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1179: "f32[512, 1024]" = torch.ops.aten.view.default(view_1178, [512, 1024]);  view_1178 = None
    permute_1025: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_276: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1179, permute_1025);  permute_1025 = None
    permute_1026: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1179, [1, 0])
    mm_277: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1026, view_24);  permute_1026 = view_24 = None
    permute_1027: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_378: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1179, [0], True);  view_1179 = None
    view_1180: "f32[1024]" = torch.ops.aten.view.default(sum_378, [1024]);  sum_378 = None
    permute_1028: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1027, [1, 0]);  permute_1027 = None
    view_1181: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_276, [1, 512, 1024]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_337: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1177, view_1181);  view_1177 = view_1181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1182: "f32[512, 1024]" = torch.ops.aten.view.default(view_1173, [512, 1024]);  view_1173 = None
    permute_1029: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_278: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
    permute_1030: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1182, [1, 0])
    mm_279: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
    permute_1031: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_379: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
    view_1183: "f32[1024]" = torch.ops.aten.view.default(sum_379, [1024]);  sum_379 = None
    permute_1032: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1031, [1, 0]);  permute_1031 = None
    view_1184: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_278, [1, 512, 1024]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_338: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_337, view_1184);  add_337 = view_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_242: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  add_9 = getitem_13 = None
    mul_859: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_242, rsqrt_2);  sub_242 = None
    mul_860: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, primals_20);  primals_20 = None
    mul_861: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_860, 1024)
    sum_380: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_860, [2], True)
    mul_862: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_860, mul_859);  mul_860 = None
    sum_381: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_862, [2], True);  mul_862 = None
    mul_863: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_859, sum_381);  sum_381 = None
    sub_243: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_861, sum_380);  mul_861 = sum_380 = None
    sub_244: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_243, mul_863);  sub_243 = mul_863 = None
    div_120: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    mul_864: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_120, sub_244);  div_120 = sub_244 = None
    mul_865: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_859);  mul_859 = None
    sum_382: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 1]);  mul_865 = None
    sum_383: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_339: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_336, mul_864);  add_336 = mul_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_70: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_866: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_867: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_339, mul_866);  mul_866 = None
    clone_139: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_867, memory_format = torch.contiguous_format);  mul_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1185: "f32[512, 1024]" = torch.ops.aten.view.default(clone_139, [512, 1024]);  clone_139 = None
    permute_1033: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_280: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
    permute_1034: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1185, [1, 0])
    mm_281: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
    permute_1035: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_384: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
    view_1186: "f32[1024]" = torch.ops.aten.view.default(sum_384, [1024]);  sum_384 = None
    permute_1036: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1035, [1, 0]);  permute_1035 = None
    view_1187: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_280, [1, 512, 4096]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_868: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf_49: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_868);  mul_868 = None
    add_340: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_869: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_340, 0.5);  add_340 = None
    mul_870: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_871: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_870, -0.5);  mul_870 = None
    exp_50: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_871);  mul_871 = None
    mul_872: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_873: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, mul_872);  view_19 = mul_872 = None
    add_341: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_869, mul_873);  mul_869 = mul_873 = None
    mul_874: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1187, add_341);  view_1187 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1188: "f32[512, 4096]" = torch.ops.aten.view.default(mul_874, [512, 4096]);  mul_874 = None
    permute_1037: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_282: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
    permute_1038: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1188, [1, 0])
    mm_283: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
    permute_1039: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_385: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
    view_1189: "f32[4096]" = torch.ops.aten.view.default(sum_385, [4096]);  sum_385 = None
    permute_1040: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1039, [1, 0]);  permute_1039 = None
    view_1190: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_282, [1, 512, 1024]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    sub_245: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  add_5 = getitem_9 = None
    mul_875: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_245, rsqrt_1);  sub_245 = None
    mul_876: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, primals_14);  primals_14 = None
    mul_877: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_876, 1024)
    sum_386: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2], True)
    mul_878: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_876, mul_875);  mul_876 = None
    sum_387: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_878, [2], True);  mul_878 = None
    mul_879: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_875, sum_387);  sum_387 = None
    sub_246: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_877, sum_386);  mul_877 = sum_386 = None
    sub_247: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_246, mul_879);  sub_246 = mul_879 = None
    div_121: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    mul_880: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_121, sub_247);  div_121 = sub_247 = None
    mul_881: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, mul_875);  mul_875 = None
    sum_388: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_881, [0, 1]);  mul_881 = None
    sum_389: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1190, [0, 1]);  view_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_342: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_339, mul_880);  add_339 = mul_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_71: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_882: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_883: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_342, mul_882);  mul_882 = None
    clone_140: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_883, memory_format = torch.contiguous_format);  mul_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1191: "f32[512, 1024]" = torch.ops.aten.view.default(clone_140, [512, 1024]);  clone_140 = None
    permute_1041: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_284: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
    permute_1042: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1191, [1, 0])
    mm_285: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
    permute_1043: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_390: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
    view_1192: "f32[1024]" = torch.ops.aten.view.default(sum_390, [1024]);  sum_390 = None
    permute_1044: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1043, [1, 0]);  permute_1043 = None
    view_1193: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_284, [1, 512, 1024]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1194: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1193, [1, 512, 16, 64]);  view_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1045: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:315, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_1195: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1045, [16, 512, 64]);  permute_1045 = None
    permute_1046: "f32[16, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_140: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1046, view_1195);  permute_1046 = None
    permute_1047: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_141: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1195, permute_1047);  view_1195 = permute_1047 = None
    view_1196: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_140, [1, 16, 512, 64]);  bmm_140 = None
    view_1197: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_141, [1, 16, 512, 512]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:309, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_72: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_884: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
    mul_885: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1197, mul_884);  view_1197 = mul_884 = None
    clone_141: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_885, memory_format = torch.contiguous_format);  mul_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:305, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_49: "f32[1, 16, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_886: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_141, alias_49);  clone_141 = None
    sum_391: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_886, [-1], True)
    mul_887: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_49, sum_391);  alias_49 = sum_391 = None
    sub_248: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_886, mul_887);  mul_886 = mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:299, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_122: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_248, 8.0);  sub_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:275, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_1198: "f32[16, 512, 512]" = torch.ops.aten.view.default(div_122, [16, 512, 512]);  div_122 = None
    permute_1048: "f32[16, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_142: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1048, view_1198);  permute_1048 = None
    permute_1049: "f32[16, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_143: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1198, permute_1049);  view_1198 = permute_1049 = None
    view_1199: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_142, [1, 16, 64, 512]);  bmm_142 = None
    view_1200: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_143, [1, 16, 512, 64]);  bmm_143 = None
    permute_1050: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1199, [0, 1, 3, 2]);  view_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1051: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1200, [0, 2, 1, 3]);  view_1200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_142: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1051, memory_format = torch.contiguous_format);  permute_1051 = None
    view_1201: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_142, [1, 512, 1024]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1052: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_1196, [0, 2, 1, 3]);  view_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_143: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1052, memory_format = torch.contiguous_format);  permute_1052 = None
    view_1202: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_143, [1, 512, 1024]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1203: "f32[512, 1024]" = torch.ops.aten.view.default(view_1202, [512, 1024]);  view_1202 = None
    permute_1053: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_286: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1203, permute_1053);  permute_1053 = None
    permute_1054: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1203, [1, 0])
    mm_287: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1054, view_5);  permute_1054 = view_5 = None
    permute_1055: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1203, [0], True);  view_1203 = None
    view_1204: "f32[1024]" = torch.ops.aten.view.default(sum_392, [1024]);  sum_392 = None
    permute_1056: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1055, [1, 0]);  permute_1055 = None
    view_1205: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_286, [1, 512, 1024]);  mm_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1057: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_1050, [0, 2, 1, 3]);  permute_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1206: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_1057, [1, 512, 1024]);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1207: "f32[512, 1024]" = torch.ops.aten.view.default(view_1206, [512, 1024]);  view_1206 = None
    permute_1058: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_288: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1207, permute_1058);  permute_1058 = None
    permute_1059: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1207, [1, 0])
    mm_289: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1059, view_2);  permute_1059 = view_2 = None
    permute_1060: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    sum_393: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True);  view_1207 = None
    view_1208: "f32[1024]" = torch.ops.aten.view.default(sum_393, [1024]);  sum_393 = None
    permute_1061: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1060, [1, 0]);  permute_1060 = None
    view_1209: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_288, [1, 512, 1024]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_343: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1205, view_1209);  view_1205 = view_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1210: "f32[512, 1024]" = torch.ops.aten.view.default(view_1201, [512, 1024]);  view_1201 = None
    permute_1062: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_290: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
    permute_1063: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1210, [1, 0])
    mm_291: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
    permute_1064: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_291, [1, 0]);  mm_291 = None
    sum_394: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
    view_1211: "f32[1024]" = torch.ops.aten.view.default(sum_394, [1024]);  sum_394 = None
    permute_1065: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1064, [1, 0]);  permute_1064 = None
    view_1212: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_290, [1, 512, 1024]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_344: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_343, view_1212);  add_343 = view_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    sub_249: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem = getitem_3 = None
    mul_888: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_249, rsqrt);  sub_249 = None
    mul_889: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, primals_4);  primals_4 = None
    mul_890: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_889, 1024)
    sum_395: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True)
    mul_891: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_889, mul_888);  mul_889 = None
    sum_396: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_891, [2], True);  mul_891 = None
    mul_892: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_888, sum_396);  sum_396 = None
    sub_250: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_890, sum_395);  mul_890 = sum_395 = None
    sub_251: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_250, mul_892);  sub_250 = mul_892 = None
    div_123: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    mul_893: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_251);  div_123 = sub_251 = None
    mul_894: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, mul_888);  mul_888 = None
    sum_397: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 1]);  mul_894 = None
    sum_398: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_345: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_342, mul_893);  add_342 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189, code: embeddings = self.dropout(embeddings)
    convert_element_type_73: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_895: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
    mul_896: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_345, mul_895);  add_345 = mul_895 = None
    clone_144: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_896, memory_format = torch.contiguous_format);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:184, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_3, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_4, clone_144);  unsqueeze_4 = scalar_tensor_4 = None
    full_6: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 1024]" = torch.ops.aten._unsafe_index_put.default(full_6, [slice_3], where_4, True);  full_6 = slice_3 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(full_1, -1)
    unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_5, scalar_tensor_5, clone_144);  unsqueeze_5 = scalar_tensor_5 = None
    full_7: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 1024]" = torch.ops.aten._unsafe_index_put.default(full_7, [full_1], where_5, True);  full_7 = full_1 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_398, 0)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_6, clone_144);  unsqueeze_6 = scalar_tensor_6 = clone_144 = None
    full_8: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[29056, 1024]" = torch.ops.aten._unsafe_index_put.default(full_8, [primals_398], where_6, True);  full_8 = primals_398 = where_6 = None
    return pytree.tree_unflatten([div_48, view_531, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_397, sum_398, permute_1065, view_1211, permute_1061, view_1208, permute_1056, view_1204, permute_1044, view_1192, sum_388, sum_389, permute_1040, view_1189, permute_1036, view_1186, sum_382, sum_383, permute_1032, view_1183, permute_1028, view_1180, permute_1023, view_1176, permute_1011, view_1164, sum_373, sum_374, permute_1007, view_1161, permute_1003, view_1158, sum_367, sum_368, permute_999, view_1155, permute_995, view_1152, permute_990, view_1148, permute_978, view_1136, sum_358, sum_359, permute_974, view_1133, permute_970, view_1130, sum_352, sum_353, permute_966, view_1127, permute_962, view_1124, permute_957, view_1120, permute_945, view_1108, sum_343, sum_344, permute_941, view_1105, permute_937, view_1102, sum_337, sum_338, permute_933, view_1099, permute_929, view_1096, permute_924, view_1092, permute_912, view_1080, sum_328, sum_329, permute_908, view_1077, permute_904, view_1074, sum_322, sum_323, permute_900, view_1071, permute_896, view_1068, permute_891, view_1064, permute_879, view_1052, sum_313, sum_314, permute_875, view_1049, permute_871, view_1046, sum_307, sum_308, permute_867, view_1043, permute_863, view_1040, permute_858, view_1036, permute_846, view_1024, sum_298, sum_299, permute_842, view_1021, permute_838, view_1018, sum_292, sum_293, permute_834, view_1015, permute_830, view_1012, permute_825, view_1008, permute_813, view_996, sum_283, sum_284, permute_809, view_993, permute_805, view_990, sum_277, sum_278, permute_801, view_987, permute_797, view_984, permute_792, view_980, permute_780, view_968, sum_268, sum_269, permute_776, view_965, permute_772, view_962, sum_262, sum_263, permute_768, view_959, permute_764, view_956, permute_759, view_952, permute_747, view_940, sum_253, sum_254, permute_743, view_937, permute_739, view_934, sum_247, sum_248, permute_735, view_931, permute_731, view_928, permute_726, view_924, permute_714, view_912, sum_238, sum_239, permute_710, view_909, permute_706, view_906, sum_232, sum_233, permute_702, view_903, permute_698, view_900, permute_693, view_896, permute_681, view_884, sum_223, sum_224, permute_677, view_881, permute_673, view_878, sum_217, sum_218, permute_669, view_875, permute_665, view_872, permute_660, view_868, permute_648, view_856, sum_208, sum_209, permute_644, view_853, permute_640, view_850, sum_202, sum_203, permute_636, view_847, permute_632, view_844, permute_627, view_840, permute_615, view_828, sum_193, sum_194, permute_611, view_825, permute_607, view_822, sum_187, sum_188, permute_603, view_819, permute_599, view_816, permute_594, view_812, permute_582, view_800, sum_178, sum_179, permute_578, view_797, permute_574, view_794, sum_172, sum_173, permute_570, view_791, permute_566, view_788, permute_561, view_784, permute_549, view_772, sum_163, sum_164, permute_545, view_769, permute_541, view_766, sum_157, sum_158, permute_537, view_763, permute_533, view_760, permute_528, view_756, permute_516, view_744, sum_148, sum_149, permute_512, view_741, permute_508, view_738, sum_142, sum_143, permute_504, view_735, permute_500, view_732, permute_495, view_728, permute_483, view_716, sum_133, sum_134, permute_479, view_713, permute_475, view_710, sum_127, sum_128, permute_471, view_707, permute_467, view_704, permute_462, view_700, permute_450, view_688, sum_118, sum_119, permute_446, view_685, permute_442, view_682, sum_112, sum_113, permute_438, view_679, permute_434, view_676, permute_429, view_672, permute_417, view_660, sum_103, sum_104, permute_413, view_657, permute_409, view_654, sum_97, sum_98, permute_405, view_651, permute_401, view_648, permute_396, view_644, permute_384, view_632, sum_88, sum_89, permute_380, view_629, permute_376, view_626, sum_82, sum_83, permute_372, view_623, permute_368, view_620, permute_363, view_616, permute_351, view_604, sum_73, sum_74, permute_347, view_601, permute_343, view_598, sum_67, sum_68, permute_339, view_595, permute_335, view_592, permute_330, view_588, permute_318, view_576, sum_58, sum_59, permute_314, view_573, permute_310, view_570, sum_52, sum_53, permute_306, view_567, permute_302, view_564, permute_297, view_560, permute_285, view_548, sum_43, sum_44, permute_281, view_545, permute_277, view_542, sum_37, sum_38, permute_273, view_539, sum_32, sum_33, permute_269, view_536, None, None, None], self._out_spec)
    